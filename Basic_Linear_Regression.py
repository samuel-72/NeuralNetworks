import tensorflow as tf
import numpy as np

# Create x and y data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Create W, b
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize error
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimzer(0.5)
train = optimizer.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Create Session
sess = tf.Session()
sess.run(init)

# Fit the line
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print (" Step No - %d\tW - %s\tb - %s\n"  % (step, sess.run(W) , sess.run(b)))
