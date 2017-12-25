import os.path
import os; os.system('cls'); os.system('clear')
import sys
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from termcolor import cprint
from tqdm import tqdm
from time import time
from glob import glob
import scipy.misc
import config
from helper import save_inference_samples
import helper

import matplotlib.pyplot as plt

test_flag = False

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Use TensorFlow version 1.0 or newer.  You have {}'.format(tf.__version__)
print('Python Version: {}'.format(sys.version))
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # https://www.tensorflow.org/api_docs/python/tf/saved_model/loader/load

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the graph from a file
    meta_graph_def = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # assign the graph to `graph`
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    cprint('VGG16 Loaded', 'blue', 'on_white')
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

if test_flag: tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # H E L P E R S
    kernel_regularizer = l2_regularizer(scale=1e-3)
    print('\n C O N V O L U T I O N')
    print('vgg_layer3_out shape: {}\t{}'.format(vgg_layer3_out.get_shape(), tf.shape(vgg_layer3_out)))
    print('vgg_layer4_out shape: {}\t{}'.format(vgg_layer4_out.get_shape(), tf.shape(vgg_layer4_out)))
    print('vgg_layer7_out shape: {}\t{}'.format(vgg_layer7_out.get_shape(), tf.shape(vgg_layer7_out)))

    # R E S A M P L E
    # we already have the 'convolution' portion from the downloaded VGG16 model
    # here, we are adding a 1x1 convolution instead of creating a fully-connected layer
    # resample vgg_layer7_out by 1x1 Convolution: To go from ?x5x18x4096 to ?x5x18x2
    layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=(1, 1),
        padding='same', kernel_regularizer=kernel_regularizer,
        name='layer7')

    # upsample vgg_layer7_out_resampled: by factor of 2 in order to go from ?x5x18x2 to ?x10x36x2
    vgg_layer7 = tf.layers.conv2d_transpose(inputs=layer7, filters=num_classes,
        kernel_size=4, strides=2, padding='same',
        kernel_regularizer=kernel_regularizer,
        name='vgg_layer7')

    # resample vgg_layer4_out out by 1x1 Convolution: To go from ?x10x36x512 to ?x10x36x2
    vgg_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=1,
        padding='same', kernel_regularizer=kernel_regularizer,
        name='vgg_layer4')

    # combined_layer1 = tf.add(vgg_layer7, vgg_layer4)\
    combined_layer1 = tf.add(vgg_layer7, vgg_layer4)

    # fcn_layer2: upsample combined_layer1 by factor of 2 in order to go from ?x10x36x2 to ?x20x72x2
    fcn_layer2 = tf.layers.conv2d_transpose(combined_layer1, num_classes,
        kernel_size=32, strides=2, padding='same',
        kernel_regularizer=kernel_regularizer,
        name='fcn_layer2')

    # resample vgg_layer3_out out by 1x1 Convolution: To go from ?x20x72x256 to ?x20x72x2
    vgg_layer3 = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=1, strides=1,
        padding='same',
        name='vgg_layer3')

    # combined_layer2 = tf.add(vgg_layer3, fcn_layer2)
    combined_layer2 = tf.add(vgg_layer3, fcn_layer2)

    # upsample combined_layer2 by factor of 8 in order to go from ?x20x72x2 to ?x160x576x2
    output = tf.layers.conv2d_transpose(combined_layer2, num_classes,
        kernel_size=32, strides=8, padding='same',
        kernel_regularizer=kernel_regularizer,
        name='output_layer')

    cprint('Layers Constructed', 'blue', 'on_white')

    return output
if test_flag: tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    softmax = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits,
        name='logits')
    cross_entropy_loss = tf.reduce_mean(softmax)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss,
        name='train_op')

    return logits, train_op, cross_entropy_loss
if test_flag: tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image, correct_label,
             keep_prob, learning_rate, logits, path_test_images):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    saver = tf.train.Saver(max_to_keep=config.models_to_keep)

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):

        start = time()
        b = 0
        for images, labels in tqdm(get_batches_fn(batch_size)):
            start_batch = time()
            sess.run(train_op, feed_dict={input_image: images,
                                                 correct_label: labels,
                                                 keep_prob: 0.5})
            b += 1
            # if b == 5: break

        '''
        M E A N  I O U

        im_softmax is list of numpy arrays.
        each array is shape (?, 2)
        each array is a softmax score to eash pixel
        '''
        # im_softmax = sess.run(
        #     [tf.nn.softmax(logits)],
        #     {keep_prob: 1.0, input_image: images})
        #
        # im_softmax = im_softmax[0]
        # print('im_softmax shape {}'.format(im_softmax.shape))
        #
        # prediction = im_softmax.reshape(160, 576, 2)
        #
        # prediction_0 = prediction[:,:,0]
        # prediction_1 = prediction[:,:,1]
        #
        # prediction_1 = prediction_1 > 0.5
        #
        # plt.imshow(prediction_0, cmap='gray'); plt.show()
        # plt.imshow(prediction_1, cmap='gray'); plt.show()

        # labels_orig = labels[0]
        # labels = labels[0].reshape(-1, 2)
        # im_softmax = im_softmax[:, 1].reshape(config.image_shape.x, config.image_shape.y)
        # im_softmax = im_softmax[:, 1].reshape(config.image_shape.x, config.image_shape.y)
        # segmentation = (im_softmax > 0.5).reshape(config.image_shape.x, config.image_shape.y, 1)
        # segmentation = (im_softmax > 0.5)
        # return Tensors for metric result and to generate results
        # below is wrong b/c we are not to use the softmax scores
        # iou, iou_op = helper.define_mean_iou(labels, segmentation, num_classes=2)
        # sess.run(tf.local_variables_initializer())
        # sess.run(iou_op)
        # cprint('MEAN IOU: {0:3.5f}'.format(sess.run(iou)), 'green', 'on_grey')

        # labels_0 = labels[:, 0]
        # labels_1 = labels[:, 1]
        #
        # labels_0 = labels_0.reshape(config.image_shape.y, config.image_shape.x)
        # labels_1 = labels_1.reshape(config.image_shape.y, config.image_shape.x)
        #
        # plt.title('label_0')
        # plt.imshow(labels_0)
        # plt.show()
        #
        # plt.title('label_1')
        # plt.imshow(labels_1)
        # plt.show()
        #
        # segmentation = segmentation[:, 1]
        # segmentation = segmentation.reshape(config.image_shape.y, config.image_shape.x)
        # plt.title('segmentation')
        # plt.imshow(segmentation)
        # plt.show()

        # S A V E  M O D E L
        cprint('EPOCH {0:2d} time --> {1:3.2f}m'.format(epoch, (time()-start)/60), 'blue', 'on_white')
        # append model name to model destination folder
        dst = os.path.join(config.model_dst, 'epoch_{:03d}'.format(epoch))
        cprint('Saving Model --> {}'.format(dst), 'blue')
        saver.save(sess, dst)

        # M E A N  I O U
        helper.compute_mean_iou(sess, logits, input_image, keep_prob)

        # create movie for finished epoch
        if epoch % config.create_movie_interval == 0:
            save_inference_samples(
                runs_dir=config.runs_dir,
                path_test_images=config.path_test_images,
                sess=sess,
                image_shape=config.image_shape_01,
                logits=logits,
                keep_prob=keep_prob,
                input_image=input_image,
                epoch=epoch)


if test_flag: tests.test_train_nn(train_nn)


def run():

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(config.data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('tensorboard')

        # Path to vgg model
        vgg_path = os.path.join(config.data_dir, 'vgg')
        # Create function to get batches

        get_batches_fn = helper.gen_batch_function(config.path_train_images,
            config.image_shape)
        images, labels = next(get_batches_fn(3))
        helper.print_data_info(images, labels)

        # # OPTIONAL: Augment Images for better results
        # #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        #
        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path=vgg_path)
        #
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, config.num_classes)
        # writer.add_graph(sess.graph)

        # O P T I M I Z E
        correct_label_holder = tf.placeholder(tf.float32,
                                              shape=(None, None, None, config.num_classes),
                                              name='correct_label_holder')
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer,
                                                        correct_label_holder,
                                                        config.LRN_RATE,
                                                        config.num_classes)

        # T R A I N
        start = time()
        os.system('clear')
        cprint('Training...', 'blue', 'on_white')
        train_nn(sess=sess,
                 epochs=config.EPOCHS,
                 batch_size=config.BATCH_SIZE,
                 get_batches_fn=get_batches_fn,
                 train_op=train_op,
                 cross_entropy_loss=cross_entropy_loss,
                 input_image=input_image,
                 correct_label=correct_label_holder,
                 keep_prob=keep_prob,
                 learning_rate=config.LRN_RATE,
                 logits=logits,
                 path_test_images=config.path_test_images)

        cprint('TOTAL time --> {0:3.2f}m'.format((time()-start)/60), 'yellow')

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
