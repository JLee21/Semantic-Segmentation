import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from termcolor import cprint


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Use TensorFlow version 1.0 or newer.  You have {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    cprint('Default GPU Device: {}'.format(tf.test.gpu_device_name()), 'blue')


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

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # conv1 = tf.nn.conv2d()
    # conv1 += biases[]
    # conv1 = tf.nn.relu(conv1)
    # pool1 = tf.nn.max_pool(conv1)
    #
    # conv2 = tf.nn.conv2d()
    # conv2 += biases
    # conv2 = tf.nn.relu(conv2)
    # pool2 = tf.nn.max_pool(conv2)
    #
    # conv3 = tf.nn.conv2d()
    # conv3 += biases
    # conv3 = tf.nn.relu(conv3)
    # pool3 = tf.nn.max_pool(conv3)
    #
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=(1,1),
        padding='same', kernel_regularizer=tf.contrib.layers.l2_reqularizer(1e-3))

    input = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2,2),
        padding='same', kernel_regularizer=tf.contrib.layers.l2_reqularizer(1e-3))

    # input = tf.add(input, pool3)

    # input = tf.add(input, pool_3)
    output = tf.layers.conv2d_transpose(input, num_classes, 16, strides=(8, 8))

    # this will print the shape after the layer is contructed?
    tf.Print(output, [tf.shape(output)])

    return nn_last_layer
# tests.test_layers(layers)


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

    softmax = tf.nn.softmax_cross_entropy_with_logits(logits, correct_label)
    cross_entropy_loss = tf.reduce_mean(softmax)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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

    for i in range(epochs):
        for images, labels in get_batches_fn(batch_size):
            loss = sess.run(train_op, feed_dict={x: images, y: labels, keep_prob: keep_prob})


# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    EPOCHS = 1
    BATCH_SIZE = 16
    LRN_RATE = 1e-3

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)


        #
        # # OPTIONAL: Augment Images for better results
        # #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        #
        # # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path=vgg_path)
        #
        nn_last_layer = layers(keep_prob, layer3_out, layer4_out, layer7_out, num_classes)
        #
        # logits, train_op, cross_entropy_loss = optimize( nn_last_layer, , LRN_RATE, num_classes=)
        #
        # # TODO: Train NN using the train_nn function
        # cprint('Training...', 'blue', 'on_white')
        # train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input,
        #          correct_label=, keep_prob, LRN_RATE)
        #
        # # TODO: Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
