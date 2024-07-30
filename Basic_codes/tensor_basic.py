import tensorflow as tf

row_dim = 0
col_dim = 0


#################--------------Fixed size tensor--------------
## Create a zero filled tensor
zero_tsr = tf.zeros([row_dim, col_dim])

## Create a ones filled tensor

one_tsr = tf.ones([row_dim, col_dim])

##Create a constant filled tensor
cont_tsr = tf.fill([row_dim,col_dim],9)

##Create a tensor out of existing constant
constant_tsr = tf.constant([1,2,3])


####################---------similar shaped tensors-------------
##Create tensors of similar shape
zeros_similar = tf.zeros_like(cont_tsr)
ones_similar = tf.ones_like(constant_tsr)


####################----------Sequence tensors------------------
#function similar to numpy's linspace which takes in the start point , end point and the  number of stops in between to be indicated
linear_tsr = tf.linspace(start = 10.0, stop = 12.0,num = 3)

#function used to generate a sequence of numbers between the start and the limit , with delta as the numbers to be generated
#the below code output -> {6,9,12}
integer_seq_tsr = tf.range(start= 6, limit= 15, delta= 3)

####################-----------Random tensors------------------

#to generate random numbers in the range of minval and maxval for uniform distributions
randunif_tsr = tf.random.uniform([row_dim,col_dim], minval = 0, maxval = 1)

#to generate random numbers from a normal distribution input bwetween mean and standard deviation , irrespective of the bounds
randnorm_tsr = tf.random.normal([row_dim,col_dim],mean = 0.0, stddev = 1.0)

#to generate the random numbers similar to above but truncated between a given range only for the normal distributions
runcnorm_tsr = tf.random.truncated_normal([row_dim,col_dim], mean = 0.0, stddev = 1.0)


###############--------------To generate random entries in an array----------

#functions used = random_shuffle() and random_crop()

input_tensor = [(2 ,3)]

shuffled_output = tf.random.shuffle(input_tensor)

cropped_output = tf.random.crop(input_tensor,crop_size = 3)

