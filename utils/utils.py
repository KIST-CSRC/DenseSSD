import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transform
import sys
import random
from PIL import Image
import numpy as np
import config


sys.path.insert(0, '..')
device = torch.device(config.device)


# Label color
CLASS_RGB = {
    'success': (0, 128, 0),
    'fail': (128, 0, 0),
    'background': ''
}

# Label class
label_class = {
    1: 'success',
    2: 'fail',
    0: 'background'
}


def xy_to_cxcy(xy):
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)


def resize(image, boxes, size=(300, 300)):
    # Resize image
    new_image = transform.resize(image, size)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims

    return new_image, new_boxes


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def compute_iou(set_1, set_2):
    # Find intersections
    intersection = find_intersection(set_1, set_2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union


def detect_objects(model, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
    """
       At certain parts, we borrow the code from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
    """
    batch_size = predicted_locs.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)
    final_images_boxes, final_images_labels, final_images_scores = [], [], []

    for i in range(batch_size):
        decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], model.priors_cxcy))

        # Lists to store boxes and scores for this image
        image_boxes, image_labels, image_scores = [], [], []

        # Check for each class
        for c in range(1, 3):
            class_scores = predicted_scores[i][:, c]
            score_above_min_score = class_scores > min_score
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]
            class_decoded_locs = decoded_locs[score_above_min_score]

            # Sort predicted boxes and scores by scores
            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
            class_decoded_locs = class_decoded_locs[sort_ind]

            # Find the overlap between predicted boxes
            overlap = compute_iou(class_decoded_locs, class_decoded_locs)

            # Non-Maximum Suppression (nms)
            nms = torch.zeros(n_above_min_score, dtype=torch.uint8).to(device)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if nms[box] == 1:
                    continue

                nms = torch.max(nms, overlap[box] > max_overlap)
                nms[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[1 - nms])
            image_labels.append(torch.LongTensor((1 - nms).sum().item() * [c]).to(device))
            image_scores.append(class_scores[1 - nms])

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        n_objects = image_scores.size(0)

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]
            image_boxes = image_boxes[sort_ind][:top_k]
            image_labels = image_labels[sort_ind][:top_k]

        # Append to lists that store predicted boxes and scores for all images
        final_images_boxes.append(image_boxes)
        final_images_labels.append(image_labels)
        final_images_scores.append(image_scores)

    return final_images_boxes, final_images_labels, final_images_scores


def random_crop(img, boxes, labels):
    if random.random() < 0.5:
        img, boxes = flip(img, boxes)

    img = np.array(img)
    boxes = torch.FloatTensor(boxes)
    labels = torch.LongTensor(labels)
    center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

    w_orig, h_orig, _ = img.shape
    h = random.uniform(0.5 * h_orig, h_orig)
    w = random.uniform(0.5 * w_orig, w_orig)
    y = random.uniform(10, h_orig - h)
    x = random.uniform(30, w_orig - w)
    h, w, x, y = int(h), int(w), int(x), int(y)

    center = center - torch.FloatTensor([[x, y]]).expand_as(center) # [n, 2]
    mask_x = (center[:, 0] >= 0) & (center[:, 0] < w) # [n,]
    mask_y = (center[:, 1] >= 0) & (center[:, 1] < h) # [n,]
    mask = (mask_x & mask_y).view(-1, 1) # [n, 1], mask for the boxes within the image after crop.

    boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4) # [m, 4]
    if len(boxes_out) == 0:
        img = Image.fromarray(img)
        return img, boxes, labels
    shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_out) # [m, 4]

    boxes_out = boxes_out - shift
    boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
    boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
    boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
    boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

    labels_out = labels[mask.view(-1)]
    img_out = img[y:y+h, x:x+w, :]
    img_out = Image.fromarray(img_out)

    if random.random() > 0.5:
        img_out, boxes_out = flip(img_out, boxes_out)

    return img_out, boxes_out, labels_out


def flip(image, boxes):
    # Flip image
    new_image = transform.hflip(image)

    boxes = np.asarray(boxes)
    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes
