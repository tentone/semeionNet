import tensorflow as tf
import semeion
import time
import image_utils

width = 16
height = 16
dataset = semeion.read_data_semeion()
classes = 10;

iterations = 1000
batch_size = 100
learning_rate = 0.001

def test(x, y):
	global prediction
	y_ = session.run(prediction, feed_dict={xs: x, keep_prob: 1})
	correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return session.run(accuracy, feed_dict={xs: x, ys: y, keep_prob: 1})

#Inputs to network
xs = tf.placeholder(tf.float32, [None, width * height]) / 255.0
ys = tf.placeholder(tf.float32, [None, classes])

keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, width, height, 1])

#Convergence layer 1
w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))#Patch 5x5, in size 1, out size 32
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

#Convergence layer 2
w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)) #patch 5x5, in size 32, out size 64
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

#Quarter image size
quarter = int(width / 4 * height / 4)

#Fully Connected layer 1
w_fc1 = tf.Variable(tf.truncated_normal([quarter * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(h_pool2, [-1, quarter * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Fully Connected layer 2
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

#Error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),  reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#Start time
start = time.time()

#Session
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

#Train the network
for i in range(iterations):
	batch_xs, batch_ys = dataset.train.next_batch(batch_size)
	session.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
	
	#Test
	if i % 50 == 0:
		print('Iteration: ' + str(i) + ' Accuracy: ' + str(test(dataset.test.images, dataset.test.labels)))

image_utils.view_weights(w_fc2.eval(session), 32, 32)

#End time
end = time.time()

#Test
print('Iteration: ' + str(i) + ' Accuracy: ' + str(test(dataset.test.images, dataset.test.labels)))

#Time
print('Time: ' + str(end - start));
