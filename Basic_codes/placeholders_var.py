import tensorflow as tf
import numpy as np

## placeholders and variables are mainly used in showing the computional graphs in tensorflow
## computational graphs are mainly meant to show the flow of data as the nodes of a directed graph

#ml -> optimize the algorithms meant for training the data
## optimization -> parameters of the algorithm 
## parameters -> variables for computational graph

###########----------To intialize and create a variable 

my_var = tf.Variable(tf.zeros([2,3]))

sess = tf.compat.v1.Session()

initialize_op = tf.compat.v1.global_variables_initializer()

sess.run(initialize_op)

##########--------------using placeholders

sess = tf.compat.v1.Session()

x = tf.compat.v1.placeholder(tf.float32,shape=[2,2])

y = tf.identity(x)

x_vals = np.random.rand(2,2)

sess.run(y,feed_dict= {x:x_vals})

## the last line will run resulting in a self referencing error 
##RuntimeError: The Session graph is empty. Add operations to the graph before calling run().


