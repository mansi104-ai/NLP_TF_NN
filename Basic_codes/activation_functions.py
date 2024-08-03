import tensorflow as tf
from tensorflow import nn

#Create a graph
graph = tf.Graph()

with graph.as_default():
    output = tf.nn.elu([-1.,0.,1.])

## Other functions : softsign, softplus, tanh, relu, relu6, sigmoid

with tf.compat.v1.Session(graph=graph) as sess:
    print(sess.run(output))