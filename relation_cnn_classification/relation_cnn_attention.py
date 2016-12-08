__author__ = 'PC-LiNing'

import datetime
import numpy
import tensorflow as tf
import dependency_load_data
# import load_data
import data_helpers
from ops import conv2d
from sklearn.metrics import recall_score,accuracy_score,f1_score
import argparse

NUM_CLASSES = 10
EMBEDDING_SIZE = 200
NUM_CHANNELS = 1
SEED = 66478
BATCH_SIZE = 128
NUM_EPOCHS = 100
EVAL_FREQUENCY = 100
META_FREQUENCY = 100
# learning rate
learning_rate_decay = 0.5
start_learning_rate = 5e-4
decay_delta = 0.05
min_learning_rate = 5e-5
# train
steps_each_check = 500
# sentence max length = 85
max_document_length = 15
window_size = 3
d_c = 100

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
    x_train = x_shuffled[1000:]
    y_train = y_shuffled[1000:]
    x_test=x_shuffled[:1000]
    y_test=y_shuffled[:1000]

    print(x_train.shape)
    print(x_test.shape)

    # input is sentence
    # [n,embed]
    train_data_node = tf.placeholder(tf.float32,shape=(max_document_length, EMBEDDING_SIZE))
    # [num_class]
    train_labels_node = tf.placeholder(tf.float32,shape=(NUM_CLASSES,))

    dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

    # convolution weight
    wf_weights = tf.Variable(
      tf.truncated_normal([d_c, EMBEDDING_SIZE],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    wf_biases = tf.Variable(tf.constant(0.01, shape=[max_document_length], dtype=tf.float32))

    # attention matrix
    u_weights = tf.Variable(
      tf.truncated_normal([d_c, d_c],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    # class embeddings matrix
    classes_matrix = tf.Variable(
      tf.truncated_normal([d_c, NUM_CLASSES],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))

    # model
    # data = [max_document_length,EMBEDDING_SIZE]
    def model(data):
        # R = [d_c,n]
        R = tf.matmul(wf_weights,data,transpose_b=True)
        # convolution_output = [d_c,n]
        convolution_output = tf.tanh(tf.nn.bias_add(R, wf_biases))
        # attention
        G_part = tf.matmul(tf.transpose(convolution_output), u_weights)
        # correlation_matrix = [n,num_class]
        correlation_matrix = tf.matmul(G_part, classes_matrix)
        # apply softmax to get attention pooling matrix
        # attention_pool = [n,num_class]
        attention_pool = tf.nn.softmax(correlation_matrix, dim=0)
        # compute output
        # W = [d_c , num_class]
        W = tf.matmul(convolution_output, attention_pool)
        # output = [d_c]
        output = tf.reduce_max(W, reduction_indices=-1)
        return output

    # score all classes
    # w_o = [d_c]
    # classes_embeddings = [num_class,d_c]
    def score_classes(w_o,classes_embeddings):
        # classes_embeddings normalized
        normalized_classes_embeddings = tf.nn.l2_normalize(classes_embeddings, dim=-1)
        all_class_embeddings = [tf.squeeze(one) for one in tf.split(0,NUM_CLASSES,normalized_classes_embeddings)]
        scores = []
        normalized_w_o = tf.nn.l2_normalize(w_o, dim=-1)
        for class_embedding in all_class_embeddings:
            scores.append(tf.nn.l2_loss(normalized_w_o - class_embedding))
        # transform to tensor
        scores = tf.pack(scores)
        return scores

    # label = [num_class],int
    # scores = [num_class],float
    # neg score is the lowest score expect true score
    def get_predict_neg_score(scores,label):
        # dot product
        ground_index = tf.argmax(label,axis=0)
        ground_score = tf.reduce_sum(tf.mul(scores, tf.cast(label, tf.float32)))
        # neg is the maximum of the remaining values
        reversed_scores = tf.negative(scores)
        top_values,top_indices = tf.nn.top_k(reversed_scores, k=2)
        true_flag = tf.nn.in_top_k(tf.expand_dims(reversed_scores,0),tf.expand_dims(ground_index,0),1)
        top_1_index = tf.cast(true_flag,tf.int32)
        chosen_score = tf.negative(tf.squeeze(tf.gather(top_values,top_1_index)))
        return ground_score, chosen_score

    def get_true_predict_indice(scores,label):
        true_indices = tf.argmax(label,axis=0)
        top_value, top_indices = tf.nn.top_k(tf.negative(scores), k=1)
        predict_indices = tf.squeeze(tf.pack(top_indices))
        return true_indices, predict_indices

    # Training computation
    w_o = model(train_data_node)
    scores = score_classes(w_o, tf.transpose(classes_matrix))
    true_score, neg_score = get_predict_neg_score(scores,train_labels_node)
    true_index, predict_index = get_true_predict_indice(scores,train_labels_node)
    # loss
    loss = true_score + 1 - neg_score
    # L2 regularization for the fully connected parameters.
    regularizers = tf.nn.l2_loss(wf_weights) + tf.nn.l2_loss(wf_biases) + tf.nn.l2_loss(u_weights) + tf.nn.l2_loss(classes_matrix)
    loss += 0.01 * regularizers

    tf.scalar_summary('loss', loss)

    # optimizer
    global_step = tf.Variable(0, name="global_step", trainable=False)
    learning_rate = tf.Variable(start_learning_rate,name="learning_rate")
    # learning_rate=tf.train.exponential_decay(start_learning_rate,global_step*BATCH_SIZE,train_size,0.9,staircase=True)
    tf.scalar_summary('lr', learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Evaluate model , 0 is wrong, 1 is right
    # train_is_correct = tf.cast(tf.equal(true_index,predict_index),tf.float32)

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
        test_size = len(x_batch)
        true_label = []
        predict_label = []
        test_loss = []
        current_step = 0
        current_lr = 0
        for i in range(test_size):
            one_feed_dict = {train_data_node: x_batch[i],train_labels_node: y_batch[i],dropout_keep_prob:1.0}
            # Run the graph and fetch some of the nodes.
            # test dont apply train_op (train_op is update gradient).
            test_step,lr,result_loss,result_true,result_predict = sess.run([global_step,learning_rate,loss,true_index,predict_index], feed_dict=one_feed_dict)
            true_label.append(result_true)
            predict_label.append(result_predict)
            test_loss.append(result_loss)
            # test_writer.add_summary(test_summary, test_step)
            current_step = test_step
            current_lr = lr

        # compute average loss
        average_loss = numpy.mean(test_loss)
        test_time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, lr {:g} ".format(test_time_str, current_step, average_loss, current_lr))
        # compute index
        compute_index(true_label,predict_label)

        new_best_test_loss = best_test_loss
        # decide if need to decay learning rate
        if (test_step % steps_each_check < 100) and (test_step > 100):
            loss_delta = (best_test_loss if best_test_loss is not None else 0 ) - average_loss
            if best_test_loss is not None and loss_delta < decay_delta:
                print('validation loss did not improve enough, decay learning rate')
                current_learning_rate = min_learning_rate if lr * learning_rate_decay < min_learning_rate else lr * learning_rate_decay
                if current_learning_rate == min_learning_rate:
                    print('It is already the smallest learning rate.')
                sess.run(learning_rate.assign(current_learning_rate))
                print('new learning rate is: ', current_learning_rate)
            else:
                # update
                new_best_test_loss = average_loss

        return new_best_test_loss

    # run the training
    with tf.Session() as sess:
        # train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',sess.graph)
        # test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
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
                if batch_count % META_FREQUENCY == 99:
                    x_batch, y_batch = zip(*batch)
                    train_size = len(x_batch)
                    true_label = []
                    predict_label = []
                    train_loss = []
                    # Run the graph and fetch some of the nodes.
                    # option
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    current_step = 0
                    for i in range(train_size):
                        feed_dict = {train_data_node: x_batch[i],train_labels_node: y_batch[i],dropout_keep_prob:0.5}
                        _,step, result_loss,result_true,result_predict = sess.run([train_op,global_step,loss,true_index,predict_index]
                                                                                          ,feed_dict=feed_dict,options=run_options,run_metadata=run_metadata)
                        true_label.append(result_true)
                        predict_label.append(result_predict)
                        train_loss.append(result_loss)
                        current_step = step
                        # train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                        # train_writer.add_summary(summary, step)

                    # compute average loss
                    average_loss = numpy.mean(train_loss)
                    acc = accuracy_score(true_label,predict_label)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g} acc {:g}".format(time_str, current_step, average_loss,acc))
                else:
                    x_batch, y_batch = zip(*batch)
                    train_size = len(x_batch)
                    true_label = []
                    predict_label = []
                    train_loss = []
                    current_step = 0
                    for i in range(train_size):
                        feed_dict = {train_data_node: x_batch[i],train_labels_node: y_batch[i],dropout_keep_prob:0.5}
                        _,step, result_loss,result_true,result_predict = sess.run([train_op,global_step,loss,true_index,predict_index],feed_dict=feed_dict)
                        true_label.append(result_true)
                        predict_label.append(result_predict)
                        train_loss.append(result_loss)
                        # train_writer.add_summary(summary, step)
                        current_step = step
                    average_loss = numpy.mean(train_loss)
                    acc = accuracy_score(true_label,predict_label)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g} acc {:g}".format(time_str,current_step, average_loss,acc))

        # train_writer.close()
        # test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, default='/tmp/cnn_attention_logs',help='Summaries directory')
    FLAGS = parser.parse_args()
    tf.app.run()