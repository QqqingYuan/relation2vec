__author__ = 'PC-LiNing'

import datetime
import argparse

import numpy

import tensorflow as tf
from test import dependency_load_data
import data_helpers

NUM_CLASSES = 10
EMBEDDING_SIZE = 100
SEED = 66478
BATCH_SIZE = 128
NUM_EPOCHS = 400
EVAL_FREQUENCY = 100
META_FREQUENCY = 100
# LSTM
# 15
max_document_length = 15
NUM_STEPS = max_document_length
num_hidden = 128
rnn_layer = 1

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

        outputs, fw_final_state, bw_final_state = tf.nn.bidirectional_rnn(fw_cell, bw_cell,x, dtype=tf.float32)

        # initial_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
        # handle  all output
        # output = [batch_size,num_hidden*2]
        merge_ouput = tf.matmul(tf.add_n(outputs), weights) + biases
        # merge_output = [batch_size,num_classes]
        return merge_ouput

    # Training computation
    # [batch_size,num_classes]
    logits = model(train_data_node,weights,biases)
    # add value clip to logits
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.clip_by_value(logits,1e-10,1.0),train_labels_node))

    tf.scalar_summary('loss', loss)

    # optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    start_learning_rate = 1e-3
    # learning_rate=tf.train.exponential_decay(start_learning_rate,global_step,5000,0.95,staircase=True)
    learning_rate = start_learning_rate

    tf.scalar_summary('lr', learning_rate)

    # adamoptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(train_labels_node,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.scalar_summary('acc', accuracy)
    merged = tf.merge_all_summaries()

    def dev_step(x_batch,y_batch,sess):
        feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:1.0}
        # Run the graph and fetch some of the nodes.
        summary,step, losses, acc= sess.run([merged,global_step, loss,accuracy],feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, lr {:g} ,acc {:g}".format(time_str, step, losses,lr,acc))
        print("{}: step {}, loss {:g} ,acc {:g}".format(time_str, step, losses,acc))

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
    parser.add_argument('--summaries_dir', type=str, default='/tmp/blstm_logs',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()