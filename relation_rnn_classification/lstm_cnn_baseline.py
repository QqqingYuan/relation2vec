__author__ = 'PC-LiNing'

import tensorflow as tf
import numpy
import time
import sys
from six.moves import xrange
import datetime
import dependency_load_data
import data_helpers
import argparse
from Highway import highways
from sklearn.metrics import recall_score,accuracy_score,f1_score

NUM_CLASSES = 10
EMBEDDING_SIZE = 100
SEED = 66478
BATCH_SIZE = 128
NUM_EPOCHS = 400
EVAL_FREQUENCY = 100
META_FREQUENCY = 100
# LSTM
# 48
max_document_length = 48
NUM_STEPS = max_document_length
num_hidden = 128
rnn_layer = 1
# CNN
NUM_CHANNELS = 1
CONVOLUTION_KERNEL_NUMBER = 100
learning_rate_decay = 0.5
# decay_delta need change when learning rate is reduce .
decay_delta = 0.01
min_learning_rate = 1e-6
start_learning_rate = 1e-3

# FLAGS=tf.app.flags.FLAGS
FLAGS = None


def train(argv=None):

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
    # 500
    steps_each_check = 500

    # input
    # input is sentence
    train_data_node = tf.placeholder(tf.float32,shape=(None,NUM_STEPS,EMBEDDING_SIZE))

    train_labels_node = tf.placeholder(tf.float32,shape=(None,NUM_CLASSES))

    dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

    weights = tf.Variable(
        tf.random_normal([2*num_hidden,NUM_CLASSES])
        # tf.truncated_normal([num_hidden,NUM_CLASSES],stddev=0.1,seed=SEED,dtype=tf.float32)
    )

    biases = tf.Variable(tf.random_normal(shape=[NUM_CLASSES], dtype=tf.float32))

    filter_sizes = [2,3,4]

    # full connected - softmax layer,
    fc1_weights = tf.Variable(
      tf.truncated_normal([CONVOLUTION_KERNEL_NUMBER * len(filter_sizes),NUM_CLASSES],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    fc1_biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES], dtype=tf.float32))

    # model
    def model(x,weights,biases):
        # Current data input shape: (batch_size, n_steps, n_input)
        x = tf.transpose(x, [1, 0, 2])
        # (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1,EMBEDDING_SIZE])
        #  get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0,NUM_STEPS, x)

        # B-directional LSTM
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=1.0,state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=1.0,state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)

        if rnn_layer > 1:
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * rnn_layer)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * rnn_layer)

        # output = [batch_size,num_hidden*2]
        # outputs of Bi-directional LSTM to highway
        outputs, fw_final_state, bw_final_state = tf.nn.bidirectional_rnn(fw_cell, bw_cell,x, dtype=tf.float32)

        # Highway
        # convert to [batch_size,num_steps,num_hidden*2]
        hw_input=tf.transpose(tf.pack(outputs,axis=0), [1, 0, 2])
        # convert to [batch_size x num_steps,num_hidden*2]
        hw_input = tf.reshape(hw_input, [-1,num_hidden*2])
        size = hw_input.get_shape()[1]
        # size = num_hidden*2
        # tf.tanh
        # hw_output=[batch_size x num_steps,num_hidden*2]
        hw_output=highways(hw_input,size)

        # convert to [batch_size,num_steps,num_hidden*2]
        hw_output = tf.reshape(hw_output, [-1,NUM_STEPS,num_hidden*2])

        # expand dim , cnn_input=[batch_size,num_steps,num_hidden*2,1]
        cnn_input=tf.expand_dims(hw_output, -1)
        # CNN
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # weights
            # one convolutional layer,have ${CONVOLUTION_KERNEL_NUMBER} kernels,and width is [3,4,5]
            conv_weights = tf.Variable(
                                tf.truncated_normal([filter_size, num_hidden*2, NUM_CHANNELS, CONVOLUTION_KERNEL_NUMBER],
                                stddev=0.1,
                                seed=SEED, dtype=tf.float32))
            conv_biases = tf.Variable(tf.zeros([CONVOLUTION_KERNEL_NUMBER], dtype=tf.float32))
            # convolution
            conv = tf.nn.conv2d(cnn_input,conv_weights,strides=[1, 1, 1, 1],padding='VALID')
            # bias and sigmoid
            tanh=tf.tanh(tf.nn.bias_add(conv, conv_biases))
            # 1-max pooling,leave a tensor of shape[batch_size,1,1,num_filters]
            pool = tf.nn.max_pool(tanh,ksize=[1,max_document_length-filter_size+1,1,1],strides=[1, 1, 1, 1],padding='VALID')
            pooled_outputs.append(pool)

        # combine all the pooled features
        num_filters_total = CONVOLUTION_KERNEL_NUMBER * len(filter_sizes)
        # [batch_size,1,1,num_filters_total]
        h_pool = tf.concat(3, pooled_outputs)
        # [batch_size,num_filters_total]
        cnn_output = tf.reshape(h_pool, [-1, num_filters_total])
        # add dropout
        cnn_output = tf.nn.dropout(cnn_output,dropout_keep_prob)
        # fc1 layer
        hidden = tf.matmul(cnn_output, fc1_weights) + fc1_biases
        return hidden

    # Training computation
    # [batch_size,num_classes]
    logits = model(train_data_node,weights,biases)
    # add value clip to logits
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.clip_by_value(logits,1e-10,1.0),train_labels_node))

    tf.scalar_summary('loss', loss)

    # optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # learning_rate=tf.train.exponential_decay(start_learning_rate,global_step,5000,0.5,staircase=True)
    learning_rate = tf.Variable(start_learning_rate,name="learning_rate")

    tf.scalar_summary('lr', learning_rate)

    # adamoptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Evaluate model
    train_predict = tf.argmax(logits,1)
    train_label = tf.argmax(train_labels_node,1)
    correct_pred = tf.equal(train_predict,train_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.scalar_summary('acc', accuracy)

    merged = tf.merge_all_summaries()

    def compute_index(y_label,y_predict):
        # macro
        print("{}: acc {:g}, recall {:g}, f1 {:g} ".format("macro",accuracy_score(y_label,y_predict),
                                                           recall_score(y_label, y_predict, average='macro'),
                                                           f1_score(y_label,y_predict, average='macro')))
        # macro
        print("{}: acc {:g}, recall {:g}, f1 {:g} ".format("micro",accuracy_score(y_label,y_predict),
                                                           recall_score(y_label, y_predict, average='micro'),
                                                           f1_score(y_label,y_predict, average='micro')))

        # weighted
        print("{}: acc {:g}, recall {:g}, f1 {:g} ".format("weighted",accuracy_score(y_label,y_predict),
                                                           recall_score(y_label, y_predict, average='weighted'),
                                                           f1_score(y_label,y_predict, average='weighted')))

    def dev_step(x_batch,y_batch,best_test_loss,sess):
        feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:0.5}
        # Run the graph and fetch some of the nodes.
        _,summary,step, losses, lr,acc,y_label,y_predict= sess.run([train_op,merged,global_step, loss,learning_rate,accuracy,train_label,train_predict]
                                                                   ,feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, lr {:g} ,acc {:g}".format(time_str, step, losses,lr,acc))
        # print("{}: step {}, loss {:g} ,acc {:g}".format(time_str, step, losses,acc))
        # compute index
        compute_index(y_label,y_predict)

        new_best_test_loss = best_test_loss
        # decide if need to decay learning rate
        if (step % steps_each_check < 100) and (step > 100):
            loss_delta = (best_test_loss if best_test_loss is not None else 0 ) - losses
            if best_test_loss is not None and loss_delta < decay_delta:
                print('validation loss did not improve enough, decay learning rate')
                current_learning_rate = min_learning_rate if lr * learning_rate_decay < min_learning_rate else lr * learning_rate_decay
                if current_learning_rate == min_learning_rate:
                    print('It is already the smallest learning rate.')
                sess.run(learning_rate.assign(current_learning_rate))
                print('new learning rate is: ', current_learning_rate)
            else:
                # update
                new_best_test_loss = losses

        return new_best_test_loss

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
        best_test_loss = None
        # Training loop.For each batch...
        for batch in batches:
            batch_count += 1
            if batch_count % EVAL_FREQUENCY == 0:
                print("\nEvaluation:")
                best_test_loss=dev_step(x_test,y_test,best_test_loss,sess)
                print("")
            else:
                if  batch_count % META_FREQUENCY == 99:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:0.5}
                    # Run the graph and fetch some of the nodes.
                    # option
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _,summary, step, losses, acc = sess.run([train_op,merged,global_step, loss,accuracy],
                                                            feed_dict=feed_dict,
                                                            options=run_options,
                                                            run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g},acc {:g}".format(time_str, step, losses,acc))
                else:
                    x_batch, y_batch = zip(*batch)
                    feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:0.5}
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
    parser.add_argument('--summaries_dir', type=str, default='/tmp/lstm_cnn_logs',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()