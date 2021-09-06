import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PIL import Image, ImageDraw, ImageFont
from model.denseSSD import denseSSD
from torchvision import transforms
from utils.utils import *
import config
import os
import cv2
import numpy as np
from natsort import natsorted


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
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 15, encoding="unic")

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


if __name__ == '__main__':
    image_dir = config.image_dir
    files = os.listdir(image_dir)
    files = natsorted(files)
    print("Detection begins!")

    # Load model checkpoint
    print("\nLoading pre-trained model!")
    model_path = config.model_path
    model = denseSSD(n_classes=config.C)
    model.load_state_dict(torch.load(model_path))
    model.to(config.device)
    model.eval()

    for file in files:
        image_path = os.path.join(image_dir, file)
        original_image = Image.open(image_path, mode='r')
        original_image = original_image.convert('RGB')

        _, objects = visualize_detection(model, original_image, min_score=0.2, max_overlap=0.5, top_k=200,
                                         path='result/'+'test_sample_'+file)
        print("Test scene file - %s: %d vials detected!" % (file, objects))

    print("\nTest completed!\n")
