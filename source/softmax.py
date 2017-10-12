import semeion
import tensorflow as tf
import image_utils
import time
import confusion

classes = 10
width = 16
height = 16
dataset = semeion.read_data_semeion()

iterations = 3000
batch_size = 100
learning_rate = 0.01

#Create the model
x = tf.placeholder(tf.float32, [None, height * width])

#Weights
W = tf.Variable(tf.zeros([height * width, classes]))

#Biases
b = tf.Variable(tf.zeros([classes]))

#Evidence
y = tf.matmul(x, W) + b

#Define loss
y_ = tf.placeholder(tf.float32, [None, classes])

#Cross-entropy
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Optimization algorithm to modify the variables and reduce the loss
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#Start time
start = time.time()

#Session
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

#Train
for i in range(iterations):
	batch_x, batch_y = dataset.train.next_batch(batch_size)
	session.run(train_step, feed_dict={x: batch_x, y_: batch_y})
	#Test
	if i % 100 == 0:
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		result = session.run(accuracy, feed_dict = {x: dataset.test.images, y_: dataset.test.labels})
		print('Iteration: ' + str(i) + ' Accuracy: ' + str(result));

#End time
end = time.time()

#Test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result = session.run(accuracy, feed_dict = {x: dataset.test.images, y_: dataset.test.labels})
print('Accuracy: ' + str(result));

#Time
print('Time: ' + str(end - start));

#Weights
image_utils.view_weights(W.eval(session), width, height)
