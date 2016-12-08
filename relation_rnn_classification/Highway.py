__author__ = 'PC-LiNing'

import tensorflow as tf


def linear(input_, output_size, scope=None):
    """
        Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
        Args:
            args: a tensor or a list of 2D, batch x n, Tensors.
            output_size: int, second dimension of W[i].
            scope: VariableScope for the created subgraph; defaults to "Linear".
        Returns:
            A 2D Tensor with shape [batch x output_size] equal to
            sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
        Raises:
            ValueError: if some of the arguments has unspecified or wrong shape.

        input_ : [batch_size,input_size]
        w :      [input_size,output_size]
        b:       [output_size]
        output:  [batch_size,output_size]

    """
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    # transpose
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


# multi-layer highway
# thanks to https://github.com/mkroutikov/tf-lstm-char-cnn
# and https://github.com/carpedm20/lstm-char-cnn-tensorflow
# x is expected 2-D arguments
def highways(x, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is non-linearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = x
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            # linear is expected 2-D arguments
            g = f(linear(x, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(x, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * x
            x = output

    return output


# TODO: highway for conv2d
def conv2d_highway(x, filter_num, filter_size,filter_length,stddev=0.1, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_size, filter_length, 1, filter_num],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
        # add batch norm
        # highways(conv,size)
        tanh = tf.tanh(conv)
        return tanh