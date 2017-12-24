import os
import sys
from collections import namedtuple

Image_Shape = namedtuple('Image_Shape', 'x y')

num_classes = 2
image_shape = (160, 576)
image_shape = Image_Shape(x=576, y=160)
EPOCHS = 1
BATCH_SIZE = 1
LRN_RATE = 1e-3
data_dir = './data'
runs_dir = './runs'
path_train_images = os.path.join(data_dir, 'data_road/training')
path_test_images = os.path.join(data_dir, 'data_road/testing/image_2/*.png')

if 'darwin' in sys.platform:
    model_dst = os.path.abspath('model')
else:
    model_dst = os.path.join('D:', 'bil/model')
