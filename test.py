from glob import glob
import os
import helper

data_dir = './data'
data_folder = 'data'
image_shape = (160, 576)
path_train_images = os.path.join(data_dir, 'data_road/training')
path_test_images = os.path.join(data_folder, 'data_road/testing/image_2/*.png')

# for image_file in glob(path_test_images):
#     print('image file ', image_file)

get_batches_fn = helper.gen_batch_function(path_train_images, image_shape)
images, labels = next(get_batches_fn(3))
helper.print_data_info(images, labels)
