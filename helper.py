import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from termcolor import cprint
# C U S T O M
from movie import create_movie
import config
# remove before submitting
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def print_data_info(images, labels):
    print('\nimages')
    print(type(images))
    print(images.shape)
    print('\nlabels')
    print(type(labels))
    print(labels.shape, '\n')

def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        cprint('Downloading pre-trained vgg model...', 'blue')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training images and labels
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        # the 'color' that represents non-drivable area in the image
        background_color = np.array([255, 0, 0])

        image_shape = (config.image_shape.y, config.image_shape.x)

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def gen_batch_function_visual_stack(data_folder, image_shape):
    """
    Similar to gen_batch_function, but this returns RGBA images for
        create_visual_stack_images
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_visual_stack_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training images and labels
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        # the 'color' that represents drivable road
        background_color = np.array([255, 0, 0])

        image_shape = (config.image_shape.y, config.image_shape.x)

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                # gt_bg = np.all(gt_image == background_color, axis=2)
                # gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                # gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_visual_stack_fn

def define_mean_iou(ground_truth, prediction, num_classes):
    """ compute the mean IOU
    """
    # print('shape of labels ', ground_truth.shape)       # (1, 160, 576, 2)
    # print('shape of prediction ', prediction.shape)  # (92160, 2)

    ground_truth = tf.convert_to_tensor(ground_truth)
    prediction = tf.convert_to_tensor(prediction)

    iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes,
        name='mean_iou')
    return iou, iou_op

def compute_mean_iou(sess, logits, input_image, keep_prob):
    """
    img.shape -> (?, 2)
    gt.shape  -> (?, 2)
    """
    get_batches_fn = gen_batch_function(config.path_train_images, config.image_shape)

    # images, labels = next(get_batches_fn(100))

    accum_mean = 0
    counter = 0
    cprint('Computing Mean_IOU', 'blue')
    for images, labels in tqdm(get_batches_fn(1)):
        counter += 1
        prediction = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, input_image: images})

        prediction = np.array(prediction)
        prediction = prediction.reshape(-1, 160, 576, 2)

        pred_thresh = prediction > 0.5
        prediction[pred_thresh] = 1

        iou, iou_op = define_mean_iou(labels, prediction, num_classes=config.num_classes)
        sess.run(tf.local_variables_initializer())
        mean_mat = sess.run(iou_op)
        mean_acc = sess.run(iou)
        accum_mean += mean_acc

        # I don't have enough GPU goodness to run all images
        if counter == config.mean_iou_counter: break

    mean_acc = accum_mean / counter
    cprint('\nMEAN IOU: {0:3.5f}'.format(mean_acc), 'green', 'on_grey')

    return mean_acc

def gen_test_output(sess, logits, keep_prob, image_pl, path_test_images, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    data_folder = config.path_train_images
    image_shape = config.image_shape
    batch_size = 1

    visual_fn = gen_batch_function_visual_stack(data_folder=data_folder, image_shape=image_shape)

    visual_gen = visual_fn(batch_size=batch_size)

    # return Tensors for metric result and to generate results
    # for image_file in glob(path_test_images):
    for i, imgs in tqdm(enumerate(visual_gen)):
        # image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        # # x == prediction image, y == ground_truth image
        x = imgs[0][0]
        y = imgs[1][0]

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [x]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape.y, image_shape.x)
        segmentation = (im_softmax > 0.5).reshape(image_shape.y, image_shape.x, 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(x)
        street_im.paste(mask, box=None, mask=mask)

        x = np.array(street_im)
        y = np.array(y)

        z = np.vstack([y, x])

        file_name = '{:03d}.png'.format(i)

        yield (file_name, z)


def plot_progress(loss, acc):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax2 = ax1.twinx()
    legends = ['loss', 'acc']
    plt.legend(legends)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross Entropy Loss')
    line1 = ax1.plot(loss, label='Loss', c='r')

    ax2.set_ylabel('Mean IOU Accuracy')
    line2 = ax2.plot(acc, label='Acc', c='g')

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.title('Training Progress')

    fig.savefig(config.progress_plot_dst)

def save_model(sess, saver, epoch, start_time):
    cprint('EPOCH {0:2d} time --> {1:3.2f}m'.format(epoch, (time.time()-start_time)/60), 'blue', 'on_white')
    dst = os.path.join(config.model_dst, 'epoch_{:03d}'.format(epoch))
    cprint('Saving Model --> {}'.format(dst), 'blue')
    saver.save(sess, dst)

def save_inference_samples(runs_dir, path_test_images, sess, image_shape,
    logits, keep_prob, input_image, epoch='na'):
    """
    1) Folder housekeeping
    2) Run NN on test images and save to Folder
    3) Create a movie/time-lapse of images saved in said Folder
    """
    # Make folder for current run
    output_dir = os.path.join(runs_dir, '-'.join(['EPOCH', str(epoch), str(int(time.time()))]))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    cprint('Training Finished. Saving test images to: {}'.format(output_dir), 'blue')
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image,
        path_test_images=path_test_images,
        image_shape=image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    # create movie from the still pngs we just created
    create_movie(path_input=output_dir,
        movie_name='-'.join(['movie','epoch', str(epoch)]))
