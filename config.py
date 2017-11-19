import os

num_classes = 2
image_shape = (160, 576)
EPOCHS = 11
BATCH_SIZE = 1
LRN_RATE = 1e-3
data_dir = './data'
runs_dir = './runs'
path_train_images = os.path.join(data_dir, 'data_road/training')
path_test_images = os.path.join(data_dir, 'data_road/testing/image_2/*.png')
