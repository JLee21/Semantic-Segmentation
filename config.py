import os
import sys
from collections import namedtuple
from termcolor import cprint

Image_Shape = namedtuple('Image_Shape', 'x y')

num_classes = 2
image_shape = Image_Shape(x=576, y=160)
EPOCHS = 5
BATCH_SIZE = 1 # GTX 1050 can only support a size of 1
batch_size_loss = 1
LRN_RATE = 1e-3
data_dir = './data'
runs_dir = './runs'
visual_dir = './visual'
path_train_images = os.path.join(data_dir, 'data_road/training')
path_test_images = os.path.join(data_dir, 'data_road/testing/image_2/*.png')
mean_iou_counter = 10
# how often to perform Inference tests and to create timelapse (epoch % create_movie_interval == 0)
create_movie_interval = 1
# how many recent models to save
models_to_keep = 2
# file path for progress plot_progress
progress_plot_dst = 'img/progress.png'
assert os.path.isdir('img')

if 'darwin' in sys.platform:
    model_dst = os.path.abspath('model')
else:
    model_dst = os.path.abspath('D:/bil/model')
    cprint('Model Destination --> {}'.format(model_dst))
