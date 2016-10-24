__author__ = 'PC-LiNing'

import tensorflow as tf


# test variable_scope

with tf.variable_scope("var1"):
    v = tf.get_variable("v", [1])

with tf.variable_scope("var1"):
    v1 = tf.get_variable("v", [1])

test1=v.assign([2])
test2=v1

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    t1,t2 = sess.run([test1,test2])
    print(t1)
    print(t2)



