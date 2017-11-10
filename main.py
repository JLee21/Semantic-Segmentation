import os.path
import os; os.system('cls'); os.system('clear')
import sys
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from termcolor import cprint
from tqdm import tqdm


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Use TensorFlow version 1.0 or newer.  You have {}'.format(tf.__version__)
print('Python Version: {}'.format(sys.version))
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

    cprint('VGG16 Loaded', 'blue', 'on_white')
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
    # H E L P E R S
    # kernel_regularizer = tf.contrib.layers.l2_reqularizer(scale=1e-3)
    print('\n C O N V O L U T I O N')
    print('vgg_layer3_out shape: {}\t{}'.format(vgg_layer3_out.get_shape(), tf.shape(vgg_layer3_out)))
    print('vgg_layer4_out shape: {}\t{}'.format(vgg_layer4_out.get_shape(), tf.shape(vgg_layer4_out)))
    print('vgg_layer7_out shape: {}\t{}'.format(vgg_layer7_out.get_shape(), tf.shape(vgg_layer7_out)))

    # do i have to do the convolutional part if i already have the layers from vgg?
    #   do i have to know the shape. do i need to change the shape?
    #   should i add biases? don't you typically add biases even in a trained model
    #   the alexnet example just takes a numpy matrix and loads those guys into
    # where is the input image specified in? where is the entry point?

    # R E S A M P L E
    # we already have the 'convolution' part from the downloaded VGG16 model
    # here, we are adding a 1x1 convolution instead of creating a fully-connected layer
    # resample vgg_layer7_out by 1x1 Convolution: To go from ?x5x18x4096 to ?x5x18x2
    layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=(1, 1), padding='same')
    print('layer7 shape: {}\t{}'.format(layer7.get_shape(), tf.shape(layer7)))
    tf.Print(layer7, [tf.shape(layer7)])

    # upsample vgg_layer7_out_resampled: by factor of 2 in order to go from ?x5x18x2 to ?x10x36x2
    vgg_layer7 = tf.layers.conv2d_transpose(layer7, num_classes, 4, 2, padding='same', name='vgg_layer7')
    print('vgg_layer7 shape: {}\t{}'.format(vgg_layer7.get_shape(), tf.shape(vgg_layer7)))

    # resample vgg_layer4_out out by 1x1 Convolution: To go from ?x10x36x512 to ?x10x36x2
    vgg_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=(1, 1), padding='same')

    # combined_layer1 = tf.add(vgg_layer7, vgg_layer4)
    combined_layer1 = tf.add(vgg_layer7, vgg_layer4)

    # fcn_layer2: upsample combined_layer1 by factor of 2 in order to go from ?x10x36x2 to ?x20x72x2
    fcn_layer2 = tf.layers.conv2d_transpose(combined_layer1, num_classes, 4, 2, padding='same', name='fcn_layer2')

    # resample vgg_layer3_out out by 1x1 Convolution: To go from ?x20x72x256 to ?x20x72x2
    vgg_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding='same')

    # combined_layer2 = tf.add(vgg_layer3, fcn_layer2)
    combined_layer2 = tf.add(vgg_layer3, fcn_layer2)

    # upsample combined_layer2 by factor of 8 in order to go from ?x20x72x2 to ?x160x576x2
    output = tf.layers.conv2d_transpose(combined_layer2, num_classes, 4, 8, padding='same', name='output_layer')

    cprint('Layers Constructed', 'blue', 'on_white')

    return output
tests.test_layers(layers)


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

    softmax = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits)
    cross_entropy_loss = tf.reduce_mean(softmax)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


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
    image_shape = (160, 576)
    x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 3), name='image_holder')
    y = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 2), name='label_holder')
    keep_prob_value = tf.constant(0.5, dtype=tf.float32)


    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(epochs):
        for images, labels in tqdm(get_batches_fn(batch_size)):
            # cprint(labels.shape, 'red')
            # print(sess.run(tf.shape(keep_prob)))

            loss = sess.run(train_op, feed_dict={input_image: images,
                                                 correct_label: labels,
                                                 keep_prob: 0.5})
        #     break
        # break

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)
    EPOCHS = 1
    BATCH_SIZE = 64
    LRN_RATE = 1e-3

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('tensorboard')


        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        images, labels = next(get_batches_fn(1))
        helper.print_data_info(images, labels)

        # # OPTIONAL: Augment Images for better results
        # #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        #
        # # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path=vgg_path)
        #
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        # writer.add_graph(sess.graph)

        # O P T I M I Z E
        correct_label_holder = tf.placeholder(tf.float32,
                                              shape=(None, None, None, num_classes),
                                              name='correct_label_holder')
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer,
                                                        correct_label_holder,
                                                        LRN_RATE,
                                                        num_classes)

        # T R A I N
        # # TODO: Train NN using the train_nn function
        cprint('Training...', 'blue', 'on_white')
        train_nn(sess=sess,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 get_batches_fn=get_batches_fn,
                 train_op=train_op,
                 cross_entropy_loss=cross_entropy_loss,
                 input_image=image_input,
                 correct_label=correct_label_holder,
                 keep_prob=keep_prob,
                 learning_rate=LRN_RATE)

        # # TODO: Save inference data using helper.save_inference_samples
        # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
