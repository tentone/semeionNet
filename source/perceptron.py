import tensorflow as tf
import semeion
import time

#Dataset
width = 16
height = 16
dataset = semeion.read_data_semeion()

#Parameters
learning_rate = 0.001
iterations = 500
batch_size = 100

#Layers number of features
n_hidden_1 = 128
n_hidden_2 = 64

#Number of input and outputs
n_input = width * height
classes = 10

#Data placeholders
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, classes])

#Layers weights
w1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
w_out = tf.Variable(tf.random_normal([n_hidden_2, classes]))

#Layers bias
b1 = tf.Variable(tf.random_normal([n_hidden_1]))
b2 = tf.Variable(tf.random_normal([n_hidden_2]))
b_out = tf.Variable(tf.random_normal([classes]))

#Hidden layer 1
layer_1 = tf.add(tf.matmul(x, w1), b1)
layer_1 = tf.nn.relu(layer_1)

#Hidden layer 2
layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
layer_2 = tf.nn.relu(layer_2)

#Output layer with linear activation
pred = tf.matmul(layer_2, w_out) + b_out

#Cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Start time
start = time.time()

#Session
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

#Training cycle
for i in range(iterations):
	total_batch = int(dataset.train.num_examples / batch_size)
	
	#Loop over all batches
	for i in range(total_batch):
		batch_x, batch_y = dataset.train.next_batch(batch_size)

		session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
	
	#Test
	if i % 5 == 0:
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("Iteration: " + str(i) + " Accuracy: ", accuracy.eval(session=session, feed_dict={x: dataset.test.images, y: dataset.test.labels}))

#End time
end = time.time()

#Test
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy:", accuracy.eval(session=session, feed_dict={x: dataset.test.images, y: dataset.test.labels}))

#Time
print('Time: ' + str(end - start));
