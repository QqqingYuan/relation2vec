__author__ = 'PC-LiNing'

import tensorflow as tf
import numpy
import argparse
import time
import sys
from six.moves import xrange
import datetime
import load_data
import data_helpers

NUM_CLASSES = 10
EMBEDDING_SIZE = 100
NUM_CHANNELS = 1
SEED = 66478
BATCH_SIZE = 128
CONVOLUTION_KERNEL_NUMBER = 100
NUM_EPOCHS = 200
EVAL_FREQUENCY = 100
META_FREQUENCY = 100

# FLAGS=tf.app.flags.FLAGS
FLAGS = None

def error_rate(predictions,labels):
    correct_predictions = tf.equal(predictions, tf.convert_to_tensor(numpy.argmax(labels, 1)))
    # accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32), name="accuracy")
    accuracy = numpy.mean(tf.cast(correct_predictions,tf.float32).eval())
    return 100*accuracy

def train(argv=None):

    # load data
    print("Loading data ... ")
    x_train,y_train = load_data.load_train_data()
    x_test,y_test = load_data.load_test_data()

    # concatenate  and shuffle .
    x_sum = numpy.concatenate((x_train,x_test))
    y_sum = numpy.concatenate((y_train,y_test))
    numpy.random.seed(10)
    shuffle_indices = numpy.random.permutation(numpy.arange(len(y_sum)))
    x_shuffled = x_sum[shuffle_indices]
    y_shuffled = y_sum[shuffle_indices]

    # split to train and test .
    x_train = x_shuffled[1000:]
    y_train = y_shuffled[1000:]
    x_test=x_shuffled[:1000]
    y_test=y_shuffled[:1000]

    print(x_train.shape)
    print(x_test.shape)

    # 85
    max_document_length = 85

    # expand (batch_size,MAX_SENTENCE_LENGTH,EMBEDDING_SIZE) to (batch_size,MAX_SENTENCE_LENGTH,EMBEDDING_SIZE,1)
    x_train = numpy.expand_dims(x_train,-1)
    x_test = numpy.expand_dims(x_test, -1)

    train_size = x_train.shape[0]
    num_epochs = NUM_EPOCHS

    filter_sizes = [3,4,5]

    # input
    # input is sentence
    train_data_node = tf.placeholder(tf.float32,shape=(None,max_document_length,EMBEDDING_SIZE,NUM_CHANNELS))

    train_labels_node = tf.placeholder(tf.float32,shape=(None,NUM_CLASSES))

    dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

    # full connected - softmax layer,
    fc1_weights = tf.Variable(
      tf.truncated_normal([CONVOLUTION_KERNEL_NUMBER * len(filter_sizes),NUM_CLASSES],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    fc1_biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES], dtype=tf.float32))

    """
    fc2_weights = tf.Variable(
      tf.truncated_normal([100,NUM_CLASSES],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES], dtype=tf.float32))
    """
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
            pool = tf.nn.max_pool(tanh,ksize=[1,max_document_length-filter_size+1,1,1],strides=[1, 1, 1, 1],padding='VALID')
            #reshape
            # pool_shape = pool.get_shape().as_list()
            # reshape = tf.reshape(pool,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            pooled_outputs.append(pool)

        # combine all the pooled features
        num_filters_total = CONVOLUTION_KERNEL_NUMBER * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # add dropout
        reshape = tf.nn.dropout(h_pool_flat,dropout_keep_prob)
        # fc1 layer
        hidden = tf.matmul(reshape, fc1_weights) + fc1_biases
        return hidden
        # if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        # return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation
    logits = model(train_data_node,True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases))
    loss += 0.05 * regularizers

    tf.scalar_summary('loss', loss)

    # optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # prediction for the current training minibatch
    # train_prediction = tf.argmax(logits, 1, name="train_predictions")
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(train_labels_node,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.scalar_summary('acc', accuracy)
    merged = tf.merge_all_summaries()

    def dev_step(x_batch,y_batch,sess):
        feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:1.0}
        # Run the graph and fetch some of the nodes.
        _, summary,step, losses, acc = sess.run([train_op,merged,global_step, loss,accuracy],feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses,acc))

    # run the training
    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

        tf.initialize_all_variables().run()
        print('Initialized!')
        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train,y_train)),BATCH_SIZE,NUM_EPOCHS)
        # batch count
        batch_count = 0
        # Training loop.For each batch...
        for batch in batches:
            batch_count += 1

            if batch_count % EVAL_FREQUENCY == 0:
                print("\nEvaluation:")
                dev_step(x_test,y_test,sess)
                print("")
            else:
                if batch_count % META_FREQUENCY == 99:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:0.75}
                    # Run the graph and fetch some of the nodes.
                    # option
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, summary, step, losses, acc = sess.run([train_op,merged,global_step, loss,accuracy],
                                                          feed_dict=feed_dict,
                                                          options=run_options,
                                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses,acc))
                else:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:0.75}
                    # Run the graph and fetch some of the nodes.
                    _, summary, step, losses, acc = sess.run([train_op,merged,global_step, loss,accuracy],feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses,acc))

        train_writer.close()
        test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='/tmp/relation_logs',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()