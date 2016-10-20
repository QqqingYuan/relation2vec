__author__ = 'PC-LiNing'

import tensorflow as tf
import numpy
import time
import sys
from six.moves import xrange
import datetime
import dependency_load_data
import data_helpers

NUM_CLASSES = 10
EMBEDDING_SIZE = 100
SEED = 66478
BATCH_SIZE = 128
NUM_EPOCHS = 200
EVAL_FREQUENCY = 100
# LSTM
# 48
max_document_length = 48
NUM_STEPS = max_document_length
num_hidden = 128
rnn_layer = 1

FLAGS=tf.app.flags.FLAGS

def main(argv=None):

    # load data
    print("Loading data ... ")
    x_train,y_train = dependency_load_data.load_train_data()
    x_test,y_test = dependency_load_data.load_test_data()

    # concatenate  and shuffle .
    x_sum = numpy.concatenate((x_train,x_test))
    y_sum = numpy.concatenate((y_train,y_test))
    numpy.random.seed(10)
    shuffle_indices = numpy.random.permutation(numpy.arange(len(y_sum)))
    x_shuffled = x_sum[shuffle_indices]
    y_shuffled = y_sum[shuffle_indices]

    # split to train and test .
    # x=[N_Samples,max_document_length,EMBEDDING_SIZE]
    # y=[N_Samples,NUM_CLASSES]
    x_train = x_shuffled[1000:]
    y_train = y_shuffled[1000:]
    x_test=x_shuffled[:1000]
    y_test=y_shuffled[:1000]

    print(x_train.shape)
    print(x_test.shape)

    train_size = x_train.shape[0]
    num_epochs = NUM_EPOCHS

    # input
    # input is sentence
    train_data_node = tf.placeholder(tf.float32,shape=(None,NUM_STEPS,EMBEDDING_SIZE))

    train_labels_node = tf.placeholder(tf.float32,shape=(None,NUM_CLASSES))

    dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

    weights = tf.Variable(
        tf.random_normal([num_hidden,NUM_CLASSES])
        # tf.truncated_normal([num_hidden,NUM_CLASSES],stddev=0.1,seed=SEED,dtype=tf.float32)
    )

    biases = tf.Variable(tf.random_normal(shape=[NUM_CLASSES], dtype=tf.float32))

    # model
    def model(x,weights,biases):
        # Current data input shape: (batch_size, n_steps, n_input)
        x = tf.transpose(x, [1, 0, 2])
        # (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1,EMBEDDING_SIZE])
        #  get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0,NUM_STEPS, x)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden,forget_bias=1.0)
        if rnn_layer > 1:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * rnn_layer)

        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

        # initial_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
        # handle  all output
        # output = [batch_size,num_hidden]
        merge_ouput = tf.matmul(tf.add_n(outputs), weights) + biases
        return merge_ouput

    # Training computation
    # [batch_size,num_classes]
    logits = model(train_data_node,weights,biases)
    # add value clip to logits
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.clip_by_value(logits,1e-10,1.0),train_labels_node))

    # optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # adamoptimizer
    # optimizer = tf.train.AdamOptimizer(1e-3)
    optimizer = tf.train.GradientDescentOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(train_labels_node,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def dev_step(x_batch,y_batch,sess):
        feed_dict = {train_data_node: x_batch,train_labels_node: y_batch}
        # Run the graph and fetch some of the nodes.
        _, step, losses, acc= sess.run([train_op,global_step, loss,accuracy],feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses,acc))

    # run the training
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print('Initialized!')
        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train,y_train)),BATCH_SIZE,NUM_EPOCHS)
        # Training loop.For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {train_data_node: x_batch,train_labels_node: y_batch}
            # Run the graph and fetch some of the nodes.
            _, step, losses, acc = sess.run([train_op,global_step, loss,accuracy],feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, losses,acc))

            if step % EVAL_FREQUENCY == 0:
                print("\nEvaluation:")
                dev_step(x_test,y_test,sess)
                print("")

if __name__ == '__main__':
  tf.app.run()