import tensorflow as tf

initializer_op = tf.compat.v1.global_variables_initializer()
#Make a session iniitializer for tensorflow
sess = tf.compat.v1.Session()


#We need to create a seperate initializer for each variable
first_var = tf.Variable(tf.zeros([2,3]))
sess.run(first_var.initializer)

#Depends on the first variable
second_var = tf.zeros_like(first_var)
sess.run(second_var.initializer)