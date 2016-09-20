__author__ = 'PC-LiNing'

import tensorflow as tf
import load_data
import numpy
import time
import sys
from six.moves import xrange

NUM_CLASSES = 10
EMBEDDING_SIZE = 100
NUM_CHANNELS = 1
SEED = 66478
BATCH_SIZE = 64
VALIDATION_SIZE = 800
CONVOLUTION_KERNEL_NUMBER = 100
EVAL_BATCH_SIZE = 128
NUM_EPOCHS = 200
EVAL_FREQUENCY = 100
MAX_SENTENCE_LENGTH = 85

FLAGS=tf.app.flags.FLAGS


def error_rate(predictions,labels):
    return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(argv=None):

    # test data
    test_data,test_label = load_data.load_test_data()
    # train data
    train_data,train_label = load_data.load_train_data()

    # expand (batch_size,MAX_SENTENCE_LENGTH,EMBEDDING_SIZE) to (batch_size,MAX_SENTENCE_LENGTH,EMBEDDING_SIZE,1)
    test_data = numpy.expand_dims(test_data,-1)
    train_data = numpy.expand_dims(train_data, -1)

    # generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE,...]
    validation_labels = train_label[:VALIDATION_SIZE,...]
    train_data = train_data[VALIDATION_SIZE:,...]
    train_labels = train_label[VALIDATION_SIZE:]
    train_size = train_data.shape[0]
    num_epochs = NUM_EPOCHS

    filter_sizes = [3,4,5]

    # input
    # input is sentence
    train_data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE,MAX_SENTENCE_LENGTH,EMBEDDING_SIZE,NUM_CHANNELS))

    train_labels_node = tf.placeholder(tf.int64,shape=(BATCH_SIZE,))

    eval_data = tf.placeholder(tf.float32,shape=(EVAL_BATCH_SIZE,MAX_SENTENCE_LENGTH,EMBEDDING_SIZE, NUM_CHANNELS))

    # full connected - softmax layer,
    fc1_weights = tf.Variable(
      tf.truncated_normal([CONVOLUTION_KERNEL_NUMBER * len(filter_sizes),NUM_CLASSES],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    fc1_biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES], dtype=tf.float32))

    # model
    def model(data,train=False):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # weights
            # one convolutional layer,have ${CONVOLUTION_KERNEL_NUMBER} kernels,and width is [3,4,5]
            conv_weights = tf.Variable(
                                tf.truncated_normal([filter_size, EMBEDDING_SIZE, NUM_CHANNELS, CONVOLUTION_KERNEL_NUMBER],
                                stddev=0.1,
                                seed=SEED, dtype=tf.float32))
            conv_biases = tf.Variable(tf.zeros([CONVOLUTION_KERNEL_NUMBER], dtype=tf.float32))
            # convolution
            conv = tf.nn.conv2d(data,conv_weights,strides=[1, 1, 1, 1],padding='VALID')
            # bias and sigmoid
            tanh=tf.tanh(tf.nn.bias_add(conv, conv_biases))
            # 1-max pooling,leave a tensor of shape[batch_size,1,1,num_filters]
            pool = tf.nn.max_pool(tanh,ksize=[1,MAX_SENTENCE_LENGTH-filter_size+1,1,1],strides=[1, 1, 1, 1],padding='VALID')
            #reshape
            # pool_shape = pool.get_shape().as_list()
            # reshape = tf.reshape(pool,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            pooled_outputs.append(pool)

        # combine all the pooled features
        num_filters_total = CONVOLUTION_KERNEL_NUMBER * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # add dropout
        reshape = tf.nn.dropout(h_pool_flat,0.5,seed=SEED)
        # fc1 layer
        hidden = tf.tanh(tf.matmul(reshape, fc1_weights) + fc1_biases)
        return hidden
        # if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        # return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation
    logits = model(train_data_node,True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases))
    loss += 0.05 * regularizers

    # optimizer
    batch = tf.Variable(0, dtype=tf.float64)
    learning_rate=tf.train.exponential_decay(0.01,batch*BATCH_SIZE,train_size,0.95,staircase=True)
    # Momentum Optimizer
    optimizer=tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)

    # prediction for the current training minibatch
    train_prediction = tf.nn.softmax(logits)

    # prediction for the test and validation.
    eval_prediction = tf.nn.softmax(model(eval_data))

    # eval in batches
    def eval_in_batches(data,sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)

        predictions = numpy.ndarray(shape=(size,NUM_CLASSES),dtype=numpy.float64)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(eval_prediction,feed_dict={eval_data: data[begin:end, ...]})
            else:
                 batch_predictions = sess.run(eval_prediction,feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                 predictions[begin:, :] = batch_predictions[begin - size:, :]

        return predictions

    # run the training
    start_time = time.time()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print('Initialized!')

        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {train_data_node: batch_data,train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction],feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %  (step, float(step) * BATCH_SIZE / train_size,1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
        # Finally print the result!
        test_error = error_rate(eval_in_batches(test_data, sess), test_label)
        print('Test error: %.1f%%' % test_error)

if __name__ == '__main__':
  tf.app.run()