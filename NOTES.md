# OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
# You'll need a GPU with at least 10 teraFLOPS to train on.
#  https://www.cityscapes-dataset.com/

# OPTIONAL: Augment Images for better results
#  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network



DICE

https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/ternaus/src/train.py

def get_dice(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1) + epsilon

    return 2 * (intersection / union).mean()

from scipy.spatial.distance import dice


tf.InteractiveSession

If you are experimenting with the programming model, and want an easy way to evaluate tensors, the tf.InteractiveSession lets you open a session at the start of your program, and then use that session for all Tensor.eval() (and Operation.run()) calls. This can be easier in an interactive setting, such as the shell or an IPython notebook, when it's tedious to pass around a Session object everywhere.
