from glob import glob
import os

data_folder = 'data'
path_test_images = os.path.join(data_folder, 'data_road/testing/image_2/*.png')

for image_file in glob(path_test_images):
    print('image file ', image_file)
