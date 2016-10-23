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

    input_ : [batch_size,hidden_size]
    w :      [hidden_size,output_size]
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
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


# one-layer highway
def highway(x, size, activation, carry_bias=-2.0):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

        t = sigmoid(W_T*x + b_T)
        y = t * g(Wx + b) + (1 - t) * x
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        the weight(W_T,W) in highway layer must have same size,but you can use fully-connected layers to change dimensionality.
        you can padding ('SAME') to maintain each layers dimensionality in convolutional layers where each layer can change the output dimensions.
    """
    W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
    b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

    W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
    b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")

    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
    H = activation(tf.matmul(x, W) + b, name="activation")
    C = tf.sub(1.0, T, name="carry_gate")
    y = tf.add(tf.mul(H, T), tf.mul(x, C), "y")
    return y


# multi-layer highway
# thanks to https://github.com/mkroutikov/tf-lstm-char-cnn
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

