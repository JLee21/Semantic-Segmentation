import matplotlib.pyplot as plt
import scipy.misc
from glob import glob

gt_path = r'C:\Users\avion105\Documents\SDC\CarND-Semantic-Segmentation\data\data_road\training\gt_image_2\*'
gt_image_files = glob(gt_path)

x = plt.imread(gt_image_files[0])
y = scipy.misc.imread(gt_image_files[0])

print(x.shape)
print(x)
