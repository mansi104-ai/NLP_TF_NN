import tensorflow as tf
import numpy as np

# Create a session
with tf.compat.v1.Session() as sess:
    # Creating matrices: either using 2D arrays or using nested loops
    identity_matrix = tf.linalg.diag([1.0, 1.0, 1.0])

    A = tf.compat.v1.truncated_normal([2, 3])
    B = tf.fill([3, 3], 5.0)
    C = tf.compat.v1.random_uniform([3, 2])
    D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., 7.]]))

    # Run the session and print the results
    # print(sess.run(identity_matrix))
    # print(sess.run(A))
    # print(sess.run(B))
    # print(sess.run(C))
    # print(sess.run(D))

    # # print(sess.run(A + B))
    # print(sess.run(B-B))

    # #print the multiplicated matrix

    # print(sess.run(tf.matmul(tf.transpose(B),identity_matrix)))
    
    # ## For the determinant:

    # print(sess.run(tf.linalg.det(D)))


    ## Inverse the matrix:
    print(sess.run(tf.linalg.inv(D)))
