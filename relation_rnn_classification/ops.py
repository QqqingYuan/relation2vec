__author__ = 'PC-LiNing'

import tensorflow as tf


def conv2d(x, filter_num, filter_size,filter_length,stddev=0.1, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_size, filter_length, 1, filter_num],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
        tanh = tf.tanh(conv)
        return tanh