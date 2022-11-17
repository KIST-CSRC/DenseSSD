import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PIL import ImageDraw, ImageFont
from torchvision import transforms
from utils.utils import *
import cv2

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def visualize_detection(model, original_image, min_score, max_overlap, top_k, path=None):

    image = normalize(to_tensor(resize(original_image))).to(device)

    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    bboxes, labels, _ = detect_objects(model, predicted_locs, predicted_scores, min_score=min_score,
                                       max_overlap=max_overlap, top_k=top_k)
    bboxes = bboxes[0].to('cpu')

    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    bboxes = bboxes * original_dims

    # Decode class integer labels
    labels = [label_class[i] for i in labels[0].to('cpu').tolist()]

    if labels == ['background']:
        return original_image, 0

    draw_label = ImageDraw.Draw(original_image)
    font = ImageFont.truetype("arial.ttf", 15, encoding="unic")

    for i in range(bboxes.size(0)):
        bboxes_coor = bboxes[i].tolist()
        draw_label.rectangle(xy=bboxes_coor, outline=CLASS_RGB[labels[i]])
        draw_label.rectangle(xy=[j + 1. for j in bboxes_coor], outline=CLASS_RGB[labels[i]])

        # Add the text
        text_size = font.getsize(labels[i].upper())
        text_coor = [bboxes_coor[0] + 2., bboxes_coor[1] - text_size[1]]
        textbox_coor = [bboxes_coor[0], bboxes_coor[1] - text_size[1],
                        bboxes_coor[2] + 1., bboxes_coor[1]]
        draw_label.rectangle(xy=textbox_coor, fill=CLASS_RGB[labels[i]])
        draw_label.text(xy=text_coor, text=labels[i].upper(), fill='white', font=font)
    del draw_label

    cv2.imwrite(path, cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2BGR))

    return original_image, bboxes.size(0)