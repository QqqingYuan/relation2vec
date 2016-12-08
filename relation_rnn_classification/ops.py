__author__ = 'PC-LiNing'

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorflow.contrib.framework.python.ops import add_arg_scope


def conv2d(x, filter_num, filter_size,filter_length,stddev=0.1, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_size, filter_length, 1, filter_num],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
        tanh = tf.tanh(conv)
        return tanh


# add batch normalization in conv2d
def batch_norm_conv2d(x, filter_num, filter_size,idx,filter_length,is_training,stddev=0.1, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [filter_size, filter_length, 1, filter_num],initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [filter_num], initializer=tf.zeros_initializer)
        # conv = [batch,x_height-filter_size+1,x_width-filter_length+1,filter_num]
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
        conv = tf.nn.bias_add(conv, b)
        # add batch norm
        conv = official_batch_norm_layer(conv,filter_num,is_training,True,scope="kernel%d_batch_norm" % idx)
        tanh = tf.tanh(conv)
        return tanh


def batch_norm_layer(inputs, phase_train, scope=None):
    return tf.cond(phase_train,
                   lambda: tf.contrib.layers.python.layers.batch_norm(inputs, is_training=True, scale=True,
                                                                      updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.python.layers.batch_norm(inputs, is_training=False, scale=True,
                                                                      updates_collections=None, scope=scope, reuse = True))


def official_batch_norm_layer(inputs,channels,phase_train,type=False,scope=None):
    return tf.cond(phase_train,
                   lambda: official_batch_norm(inputs,channels,type,is_training=True, scale=True,
                                               updates_collections=None, scope=scope),
                   lambda: official_batch_norm(inputs,channels,type,is_training=False, scale=True,
                                               updates_collections=None, scope=scope, reuse=True))


def batch_norm(x, size,name_scope, training, epsilon=1e-3, decay=0.999,train=True):
    """
        Assume 2d [batch, values] tensor
        size = values
    """
    with tf.variable_scope(name_scope):
        # size = x.get_shape().as_list()[1]
        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1),trainable=train)
        offset = tf.get_variable('offset', [size],initializer=tf.zeros_initializer,trainable=train)

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


def convolution_batch_norm(x, channels,name_scope, training, epsilon=1e-3, decay=0.999,train=True):
    """
        Assume 4d [batch, height,width,channels] tensor
    """
    with tf.variable_scope(name_scope):
        # size = x.get_shape().as_list()[-1]
        # scale (gamma) init to 0.1
        scale = tf.get_variable('scale', [channels], initializer=tf.constant_initializer(0.1),trainable=train)
        offset = tf.get_variable('offset', [channels],initializer=tf.zeros_initializer,trainable=train)

        pop_mean = tf.get_variable('pop_mean', [channels], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [channels], initializer=tf.ones_initializer, trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


@add_arg_scope
def official_batch_norm(inputs,channels,type=False,decay=0.999,center=True,scale=False,epsilon=0.001,activation_fn=None,
                        updates_collections=ops.GraphKeys.UPDATE_OPS,is_training=True,reuse=None,variables_collections=None,
                        outputs_collections=None,trainable=True,scope=None):
    """
        Args:
            inputs: a tensor of size `[batch_size, height, width, channels]`
                or `[batch_size, channels]`.
            type: False is non-convolution batch norm,True is convolution batch norm.
            decay: decay for the moving average.
            center: If True, subtract `beta`. If False, `beta` is ignored.
            scale: If True, multiply by `gamma`. If False, `gamma` is
                not used. When the next layer is linear (also e.g. `nn.relu`), this can be
                disabled since the scaling can be done by the next layer.
            epsilon: small float added to variance to avoid dividing by zero.
            activation_fn: Optional activation function.
            updates_collections: collections to collect the update ops for computation.
            is_training: whether or not the layer is in training mode.
            reuse: whether or not the layer and its variables should be reused.
            variables_collections: optional collections for the variables.
            outputs_collections: collections to add the outputs.
            trainable: If `True` also add variables to the graph collection
                `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
            scope: Optional scope for `variable_op_scope`.
            Returns:
                a tensor representing the output of the operation.
    """
    with variable_scope.variable_op_scope([inputs],scope, 'BatchNorm', reuse=reuse) as sc:
        dtype = tf.float32
        axis = [0,1,2] if type else [0]
        params_shape = [channels]
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta_collections = utils.get_variable_collections(variables_collections,'beta')
            beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=init_ops.zeros_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
        if scale:
            gamma_collections = utils.get_variable_collections(variables_collections,'gamma')
            gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=init_ops.ones_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections.
        moving_mean_collections = utils.get_variable_collections(variables_collections, 'moving_mean')
        moving_mean = variables.model_variable('moving_mean',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=init_ops.zeros_initializer,
                                             trainable=False,
                                             collections=moving_mean_collections)
        moving_variance_collections = utils.get_variable_collections(variables_collections, 'moving_variance')
        moving_variance = variables.model_variable('moving_variance',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=init_ops.ones_initializer,
                                             trainable=False,
                                             collections=moving_variance_collections)
        if is_training:
            # Calculate the moments based on the individual batch.
            mean, variance = nn.moments(inputs, axis, shift=moving_mean)
            # Update the moving_mean and moving_variance moments.
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            if updates_collections is None:
                # Make sure the updates are computed here.
                with ops.control_dependencies([update_moving_mean,update_moving_variance]):
                    outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
            else:
                # Collect the updates to be computed later.
                ops.add_to_collections(updates_collections, update_moving_mean)
                ops.add_to_collections(updates_collections, update_moving_variance)
                outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        else:
            outputs = nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon)
            # TODO:shape
            # outputs.set_shape(inputs.get_shape())
        if activation_fn:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)