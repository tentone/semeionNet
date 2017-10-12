import tensorflow as tf
import semeion
import time
from tensorflow.contrib import rnn

width = 16
height = 16
dataset = semeion.read_data_semeion()
classes = 10

#Parameters
learning_rate = 0.01
iterations = 100000
batch_size = 100
display_step = 2
forget_bias = 1.0

#Network Parameters
inputs = width
steps = height
features = 128

#Data placeholders
x = tf.placeholder('float', [None, steps, inputs])
y = tf.placeholder('float', [None, classes])

#Weights
weights = tf.Variable(tf.random_normal([features, classes]))

#Biases
biases = tf.Variable(tf.random_normal([classes]))

#Unstack to get a list of steps tensors of shape (batch_size, inputs)
ux = tf.unstack(x, steps, 1)

#LSTM cell
lstm_cell = rnn.BasicLSTMCell(features, forget_bias=forget_bias)

#RNN cell
outputs, states = rnn.static_rnn(lstm_cell, ux, dtype=tf.float32)

#Linear activation, using rnn inner loop last output
prediction = tf.matmul(outputs[-1], weights) + biases

#Cost
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#Evaluate model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Start time
start = time.time()

#Session
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

step = 1

#Train
while step * batch_size < iterations:
	batch_x, batch_y = dataset.train.next_batch(batch_size)	
	batch_x = batch_x.reshape((batch_size, steps, inputs))

	session.run([optimizer, cross_entropy], feed_dict = {x: batch_x, y: batch_y})
	
	#Test
	if step % display_step == 0:
		test_data = dataset.test.images[:].reshape((-1, steps, inputs))
		result = session.run(accuracy, feed_dict={x: test_data, y: dataset.test.labels})
		print('Iteration: ' + str(step * batch_size) + ' Accuracy: ' + str(result))
	step += 1

#End time
end = time.time()

#Test
test_data = dataset.test.images[:].reshape((-1, steps, inputs))
result = session.run(accuracy, feed_dict={x: test_data, y: dataset.test.labels})
print('Iteration: ' + str(step * batch_size) + ' Accuracy: ' + str(result))

#Time
print('Time: ' + str(end - start));