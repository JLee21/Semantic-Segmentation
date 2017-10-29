import tensorflow as tf
import sys
from termcolor import cprint

cprint('Python Version: {}'.format(sys.version), 'blue', 'on_white')
cprint('Tensorflow Version: {}'.format(tf.__version__), 'blue', 'on_white')

hello = tf.constant('Hello')
sess = tf.Session()
print(sess.run(hello))
