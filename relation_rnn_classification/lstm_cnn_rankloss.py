__author__ = 'PC-LiNing'

import datetime
import argparse

import numpy
from sklearn.metrics import recall_score,accuracy_score,f1_score

import tensorflow as tf
import dependency_load_data
import data_helpers
from Highway import highways
from ops import conv2d

NUM_CLASSES = 10
EMBEDDING_SIZE = 200
SEED = 66478
BATCH_SIZE = 128
NUM_EPOCHS = 200
EVAL_FREQUENCY = 100
META_FREQUENCY = 100
# LSTM
# 15
max_document_length = 15
NUM_STEPS = max_document_length
num_hidden = 256
rnn_layer = 1
# CNN
NUM_CHANNELS = 1
learning_rate_decay = 0.5
# decay_delta need change when learning rate is reduce .
decay_delta = 0.1
min_learning_rate = 5e-5
start_learning_rate = 1e-3

# test size
Test_Size = 733

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
    x_train = x_shuffled[Test_Size:]
    y_train = y_shuffled[Test_Size:]
    x_test=x_shuffled[:Test_Size]
    y_test=y_shuffled[:Test_Size]

    print(x_train.shape)
    print(x_test.shape)
    print("exception words : "+str(dependency_load_data.get_exception_number()))
    # 500
    steps_each_check = 500

    # input
    # input is sentence
    train_data_node = tf.placeholder(tf.float32,shape=(None,NUM_STEPS,EMBEDDING_SIZE))

    train_labels_node = tf.placeholder(tf.float32,shape=(None,NUM_CLASSES))

    dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

    filter_sizes = [1,2,3,4,5,6]
    filter_numbers = [300,300,200,200,150,150]
    # full connected - softmax layer,
    fc1_weights = tf.Variable(
      tf.truncated_normal([sum(filter_numbers),200],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    fc1_biases = tf.Variable(tf.constant(0.01, shape=[200], dtype=tf.float32))

    fc2_weights = tf.Variable(
      tf.truncated_normal([200,NUM_CLASSES],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    fc2_biases = tf.Variable(tf.constant(0.01, shape=[NUM_CLASSES], dtype=tf.float32))

    # model
    def model(x):
        # Current data input shape: (batch_size, n_steps, n_input)
        x = tf.transpose(x, [1, 0, 2])
        # (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1,EMBEDDING_SIZE])
        #  get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0,NUM_STEPS, x)

        # B-directional LSTM
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=1.0,state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,forget_bias=1.0,state_is_tuple=True)

        if rnn_layer > 1:
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * rnn_layer)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * rnn_layer)

        # output = [batch_size,num_hidden*2]
        # outputs of Bi-directional LSTM to highway
        outputs, fw_final_state, bw_final_state = tf.nn.bidirectional_rnn(fw_cell, bw_cell,x, dtype=tf.float32)

        # Highway
        # convert to [batch_size,num_steps,num_hidden*2]
        hw_input=tf.transpose(tf.pack(outputs,axis=0), [1, 0, 2])

        # add dropout for Bi-LSTM output
        hw_input = tf.nn.dropout(hw_input,dropout_keep_prob)

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
        for idx, filter_size in enumerate(filter_sizes):
            conv = conv2d(cnn_input,filter_numbers[idx],filter_size,num_hidden*2,name="kernel%d" % idx)
            # 1-max pooling,leave a tensor of shape[batch_size,1,1,num_filters]
            pool = tf.nn.max_pool(conv,ksize=[1,max_document_length-filter_size+1,1,1],strides=[1, 1, 1, 1],padding='VALID')
            pooled_outputs.append(tf.squeeze(pool))

        if len(filter_sizes) > 1:
            cnn_output = tf.concat(1,pooled_outputs)
        else:
            cnn_output = pooled_outputs[0]

        # add dropout
        cnn_output = tf.nn.dropout(cnn_output,dropout_keep_prob)
        # fc1 layer
        hidden = tf.matmul(cnn_output, fc1_weights) + fc1_biases
        fc1_output = tf.sigmoid(hidden)
        # fc2 layer
        fc_output = tf.matmul(fc1_output,fc2_weights) + fc2_biases
        return fc_output

    # Training computation
    # [batch_size,num_classes]
    logits = model(train_data_node)
    logits = tf.nn.softmax(logits)
    train_predict = tf.argmax(logits,1)
    train_label = tf.argmax(train_labels_node,1)

    def get_targe_neg(size):
        # Evaluate model
        current_batch_size = size
        true_index = tf.cast(train_label,tf.int32)
        true_idx_flattened = tf.range(0,current_batch_size) * NUM_CLASSES + true_index
        true_values = tf.gather(tf.reshape(logits,[-1]),true_idx_flattened)
        top_2_values, top_2_indices = tf.nn.top_k(logits,2)
        top_1_flag = tf.nn.in_top_k(logits,true_index,1)
        top_1_index = tf.cast(top_1_flag,tf.int32)
        rows = [ tf.squeeze(row) for row in tf.split(0,current_batch_size,top_2_indices)]
        idxs = tf.split(0,current_batch_size,top_1_index)
        neg_idx = [ tf.gather(rows[i],idxs[i]) for i in range(current_batch_size)]
        neg_idx = tf.concat(0, neg_idx)
        neg_idx = tf.range(0,current_batch_size) * NUM_CLASSES + neg_idx
        neg_values = tf.gather(tf.reshape(logits,[-1]),neg_idx)
        return true_values,neg_values

    is_train = tf.constant(0.5)
    t_values , n_values = tf.cond(tf.equal(dropout_keep_prob,is_train),lambda: get_targe_neg(BATCH_SIZE),lambda: get_targe_neg(Test_Size))
    # add value clip to logits
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.clip_by_value(logits,1e-10,1e10),train_labels_node))
    # pairwise ranking loss
    loss = tf.reduce_mean(1.0 - t_values + n_values)
    """
    m_target = 0.5
    m_neg = 0.0
    r = 5
    loss = tf.reduce_mean(tf.log(1+tf.exp(r*(m_target-t_values)))+tf.log(1+tf.exp(r*(m_neg+n_values))))
    """
    regularization = tf.nn.l2_loss(fc1_weights)+tf.nn.l2_loss(fc1_biases)+tf.nn.l2_loss(fc2_weights)\
                     + tf.nn.l2_loss(fc2_biases)
    loss += 0.01 * regularization

    tf.scalar_summary('loss', loss)

    # optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # learning_rate=tf.train.exponential_decay(start_learning_rate,global_step,5000,0.5,staircase=True)
    learning_rate = tf.Variable(start_learning_rate,name="learning_rate")

    tf.scalar_summary('lr', learning_rate)

    # adam-optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # train accuracy
    train_correct_pred = tf.equal(train_predict,train_label)
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))
    tf.scalar_summary('acc', train_accuracy)
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
        feed_dict = {train_data_node: x_batch,train_labels_node: y_batch,dropout_keep_prob:1.0}
        # Run the graph and fetch some of the nodes.
        # test dont apply train_op (train_op is update gradient).
        summary,step, losses, lr, acc, y_label,y_predict = sess.run([merged,global_step, loss,learning_rate,train_accuracy,train_label,train_predict], feed_dict=feed_dict)
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
                    _, summary, step, losses, acc = sess.run([train_op,merged,global_step, loss,train_accuracy],
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
                    _, summary, step, losses ,acc = sess.run([train_op,merged,global_step, loss, train_accuracy],feed_dict=feed_dict)
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