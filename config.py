# GPU Device
device = "cuda:0"

# Path to data dir
test_sample_image_dir = 'dataset/test_sample'
test_result_image_dir = 'dataset/test_result'
image_dir_train = 'dataset/training_sample/Empty/img'
info_dir_train  = 'dataset/training_sample/Empty/annotation'
train_label     = 'train.txt'
val_label       = 'valid.txt'

# Hyper parameters
model_path = 'pre-trained/model.pth'
image_size = 300
C = 3
batch_size      = 1
init_lr         = 0.001
weight_decay    = 5e-4
num_epochs      = 5