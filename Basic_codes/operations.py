import tensorflow as tf

#Create a graph
graph = tf.Graph()

#Declare oprations to perform the required function 
with graph.as_default() :

    #What to be included in the graph
    def custom_polynomial(value):
        return(tf.math.subtract(3* tf.square(value),value) +10)


# A session is always run in a computational graph as an instance 
with tf.compat.v1.Session(graph=graph) as sess:

    # result = quotient
    #running the session for the graph.

    print(sess.run(custom_polynomial(1)))

