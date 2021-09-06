# Vial-positioning Detection

### Introduction
This work contributes vial-positioning detection (Dense SSD) in autonomous system laboratory of KIST.

We also provide the codes as follows:
  1) Dense SSD (see [model](https://github.com/KIST-CSRC/vial-positioning-detection/tree/master/model))
  2) Test code (see [detect.py](https://github.com/KIST-CSRC/vial-positioning-detection/detect.py))

### Requirements
  1) [PyTorch](https://pytorch.org/)
  2) [PIL](https://pillow.readthedocs.io/en/stable/installation.html)
  3) [NatSort](https://pypi.org/project/natsort/)
  4) [OpenCV](https://pypi.org/project/opencv-python/)

### Pre-trained Model
Please ensure that after unzipping the pre-trained folder and set up the directories as follows:
```
Vial-positioning
├── dataset
│   └── test_sample
├── model
│   └── denseSSD.py
├── pre-trained
│   └── model.pth
├── utils
│   └── utils.py
├── config.py
└── detect.py
```

### Compatibility
We tested the codes with:
  1) PyTorch 1.7.0 under Ubuntu OS 16.04/18.04 LTS and Anaconda3 (Python 3.7)
  2) PyTorch 1.7.0 under Windows 10 and Anaconda3 (Python 3.7)

