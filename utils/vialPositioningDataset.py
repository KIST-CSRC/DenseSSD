from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from utils.utils import *
import torchvision.transforms.functional as transform
import torch
import os
import numpy
import cv2 as cv


class vialPositioningDataset(Dataset):
    def __init__(self, image_dir, label_txt, image_size=300, training=False, debug=False):
        self.image_size = image_size
        self.debug = debug
        self.paths = []
        self.labels = []
        self.bboxes = []
        self.difficulties = []
        self.training = training

        with open(label_txt) as f:
            lines = f.readlines()

        for line in lines:
            temp_split = line.strip().split()
            file_name = temp_split[0]
            path = os.path.join(image_dir, file_name)

            # Ignore the missing images
            if not os.path.exists(path):
                continue

            self.paths.append(path)

            # Get the bounding boxes info
            num_boxes = (len(temp_split) - 1) // 5
            bbox, label, difficulties = [], [], []

            for i in range(num_boxes):
                x1 = float(temp_split[5 * i + 1])
                y1 = float(temp_split[5 * i + 2])
                x2 = x1 + float(temp_split[5 * i + 3])
                y2 = y1 + float(temp_split[5 * i + 4])
                c = float(temp_split[5 * i + 5])
                bbox.append([x1, y1, x2, y2])
                label.append(c)
                difficulties.append(c)
            self.labels.append(label)
            self.bboxes.append(bbox)
            self.difficulties.append(1)

        self.num_samples = len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx], mode='r')
        image = image.convert('RGB')

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        bboxes = self.bboxes[idx]
        labels = self.labels[idx]
        difficulties = self.difficulties[idx]

        if self.training is True:
            num_rand = random.random()
            if num_rand < 0.35:
                image, bboxes = flip(image, bboxes)
            if 0.35 <= num_rand < 0.75:
                image, bboxes, labels = random_crop(image, bboxes, labels)

        # For debugging the labels info
        if self.debug is True:
            debug_dir = 'tmp/check'
            os.makedirs(debug_dir, exist_ok=True)
            img_show = numpy.asarray(image)
            box_show = numpy.asarray(bboxes).reshape(-1)
            n = len(box_show) // 4
            CLASS_BGR = {
                'fail': (0, 0, 128),
                'success': (0, 128, 0)
            }
            name_bgr_dict = CLASS_BGR
            for b in range(n):
                pt1 = (int(box_show[4 * b + 0]), int(box_show[4 * b + 1]))
                pt2 = (int(box_show[4 * b + 2]), int(box_show[4 * b + 3]))

                class_name = labels[b]
                if int(class_name) == 1:
                    class_name = "success"
                else:
                    class_name = "fail"
                size, baseline = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
                text_w, text_h = size

                x = int(box_show[4 * b + 0])
                y = int(box_show[4 * b + 1])

                x1y1 = (x, y)
                x2y2 = (x + text_w + 1, y + text_h + 1 + baseline)
                bgr = name_bgr_dict[class_name]
                cv.rectangle(img_show, pt1, pt2, bgr, thickness=1)
                cv.rectangle(img_show, x1y1, x2y2, bgr, -1)
                cv.putText(img_show, class_name, (x + 1, y + 2 * baseline + 1),
                           cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)
            cv.imwrite(os.path.join(debug_dir, 'test_{}.jpg'.format(idx)), img_show)

        bboxes = torch.FloatTensor(bboxes)
        labels = torch.LongTensor(labels)
        difficulties = torch.ByteTensor(difficulties)

        new_image, new_bboxes = resize(image, bboxes, size=(self.image_size, self.image_size))
        new_image = transform.to_tensor(new_image)

        new_image = transform.normalize(new_image, mean=mean, std=std)

        return new_image, new_bboxes, labels, difficulties

    def __len__(self):
        return self.num_samples

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties
