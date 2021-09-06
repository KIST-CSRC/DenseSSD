import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import math
import config


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class Dimension_Reduction(nn.Module):
    def __init__(self, in_planes, out_planes, ceil_mode=True):
        super(Dimension_Reduction, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.ceil_mode = ceil_mode

    def forward(self, x):
        out = F.avg_pool2d(x, 2, ceil_mode=self.ceil_mode)
        out = self.conv(F.relu(self.bn(out)))

        return out


class denseNet(nn.Module):
    def __init__(self, block):
        super(denseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.dense1 = self._make_dense_layers(block, 64, 6)
        self.trans1 = Transition(256, 256)
        self.dense2 = self._make_dense_layers(block, 256, 8)
        self.trans2 = Transition(512, 512)
        self.dense3 = self._make_dense_layers(block, 512, 16)
        self.trans3 = Transition(1024, int(math.floor(1024*0.5)))
        self.dr1 = Dimension_Reduction(512, 512)
        self.dense4 = self._make_dense_layers(block, 512, 8)
        self.trans4 = Transition(768, int(math.floor(1280*0.2)))
        self.dr2 = Dimension_Reduction(1024, 256)
        self.conv_block4 = nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1)
        self.dr3 = Dimension_Reduction(512, 128)
        self.conv_block5 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.dr4 = Dimension_Reduction(256, 128)
        self.conv_block6 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.dr5 = Dimension_Reduction(256, 128)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)

        out = self.dense2(out)
        out1_block = self.trans2(out)

        out = self.dense3(out1_block)
        out = self.trans3(out)
        out1 = self.dr1(out1_block)
        out2_block = torch.cat([out1, out], 1)

        out = self.dense4(out)
        out = self.trans4(out)
        out2 = self.dr2(out2_block)
        out3_block = torch.cat([out2, out], 1)

        out3 = self.dr3(out3_block)
        out = self.conv_block4(out3_block)
        out4_block = torch.cat([out3, out], 1)

        out4 = self.dr4(out4_block)
        out = self.conv_block5(out4_block)
        out5_block = torch.cat([out4, out], 1)

        out5 = self.dr5(out5_block)
        out = self.conv_block6(out5_block)
        out6_block = F.max_pool2d(torch.cat([out5, out], 1), 2)

        return out1_block, out2_block, out3_block, out4_block, out5_block, out6_block

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, 32))
            in_planes += 32
        return nn.Sequential(*layers)


class PredictionLayer(nn.Module):
    def __init__(self, n_classes):
        super(PredictionLayer, self).__init__()
        self.C = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'1': 4, '2': 6, '3': 6, '4': 6, '5': 4, '6': 4}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_1 = nn.Conv2d(512, n_boxes['1'] * 4, kernel_size=3, padding=1)
        self.loc_2 = nn.Conv2d(1024, n_boxes['2'] * 4, kernel_size=3, padding=1)
        self.loc_3 = nn.Conv2d(512, n_boxes['3'] * 4, kernel_size=3, padding=1)
        self.loc_4 = nn.Conv2d(256, n_boxes['4'] * 4, kernel_size=3, padding=1)
        self.loc_5 = nn.Conv2d(256, n_boxes['5'] * 4, kernel_size=3, padding=1)
        self.loc_6 = nn.Conv2d(256, n_boxes['6'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cls_1 = nn.Conv2d(512, n_boxes['1'] * n_classes, kernel_size=3, padding=1)
        self.cls_2 = nn.Conv2d(1024, n_boxes['2'] * n_classes, kernel_size=3, padding=1)
        self.cls_3 = nn.Conv2d(512, n_boxes['3'] * n_classes, kernel_size=3, padding=1)
        self.cls_4 = nn.Conv2d(256, n_boxes['4'] * n_classes, kernel_size=3, padding=1)
        self.cls_5 = nn.Conv2d(256, n_boxes['5'] * n_classes, kernel_size=3, padding=1)
        self.cls_6 = nn.Conv2d(256, n_boxes['6'] * n_classes, kernel_size=3, padding=1)

    def forward(self, y1, y2, y3, y4, y5, y6):
        batch_size = y1.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        loc_1 = self.loc_1(y1)
        loc_1 = loc_1.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        loc_2 = self.loc_2(y2)
        loc_2 = loc_2.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        loc_3 = self.loc_3(y3)
        loc_3 = loc_3.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        loc_4 = self.loc_4(y4)
        loc_4 = loc_4.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        loc_5 = self.loc_5(y5)
        loc_5 = loc_5.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        loc_6 = self.loc_5(y6)
        loc_6 = loc_6.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        # Predict classes in localization boxes
        c_1 = self.cls_1(y1)
        c_1 = c_1.permute(0, 2, 3, 1).reshape(batch_size, -1, self.C)

        c_2 = self.cls_2(y2)
        c_2 = c_2.permute(0, 2, 3, 1).reshape(batch_size, -1, self.C)

        c_3 = self.cls_3(y3)
        c_3 = c_3.permute(0, 2, 3, 1).reshape(batch_size, -1, self.C)

        c_4 = self.cls_4(y4)
        c_4 = c_4.permute(0, 2, 3, 1).reshape(batch_size, -1, self.C)

        c_5 = self.cls_5(y5)
        c_5 = c_5.permute(0, 2, 3, 1).reshape(batch_size, -1, self.C)

        c_6 = self.cls_6(y6)
        c_6 = c_6.permute(0, 2, 3, 1).reshape(batch_size, -1, self.C)

        locs = torch.cat([loc_1, loc_2, loc_3, loc_4, loc_5, loc_6], dim=1)
        classes_scores = torch.cat([c_1, c_2, c_3, c_4, c_5, c_6], dim=1)

        return locs, classes_scores


class denseSSD(nn.Module):
    def __init__(self, n_classes):
        super(denseSSD, self).__init__()

        self.n_classes = n_classes

        self.denseNet = denseNet(Bottleneck)
        self.pred_layers = PredictionLayer(n_classes)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        y1, y2, y3, y4, y5, y6 = self.denseNet(image)
        blocs, classes_scores = self.pred_layers(y1, y2, y3, y4, y5, y6)

        return blocs, classes_scores

    def create_prior_boxes(self):
        fmap_dims = {'1': 38, '2': 19, '3': 10, '4': 5, '5': 3, '6': 1}

        obj_scales = {'1': 0.1, '2': 0.2, '3': 0.375, '4': 0.55, '5': 0.725, '6': 0.9}

        aspect_ratios = {'1': [1., 2., 0.5],
                         '2': [1., 2., 3., 0.5, .333],
                         '3': [1., 2., 3., 0.5, .333],
                         '4': [1., 2., 3., 0.5, .333],
                         '5': [1., 2., 0.5],
                         '6': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())
        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        if ratio == 1.:
                            try:
                                additional_scale = math.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(config.device)
        prior_boxes.clamp_(0, 1)

        return prior_boxes
