import numpy as np
import tensorflow as tf
import semeion
import time

width = 16
height = 16
dataset = semeion.read_data_semeion()
classes = 10

#Training data
Xtr, Ytr = dataset.train.next_batch(99999999)

#Testing data
Xte, Yte = dataset.test.next_batch(99999999)

#Graph Input
xtr = tf.placeholder('float', [None, width * height])
xte = tf.placeholder('float', [width * height])

#Nearest Neighbor calculating distance
distance = tf.reduce_sum(tf.abs(tf.subtract(xtr, xte)), reduction_indices = 1)

#Prediction
prediction = tf.arg_min(distance, 0)

#Start time
start = time.time()

#Session
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

accuracy = 0.0

#Test
for i in range(len(Xte)):
	nn_index = session.run(prediction, feed_dict={xtr: Xtr, xte: Xte[i, :]})

	if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
		accuracy += 1.0 / len(Xte)

#End time
end = time.time()

print('Time: ' + str(end - start));
print('Accuracy:' + str(accuracy))