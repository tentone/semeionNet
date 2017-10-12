import collections
import numpy
import dataset
import image_utils
from random import shuffle
from tensorflow.python.framework import dtypes

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def read_data_semeion(fname = 'dataset/semeion/semeion.data'):
	file = open(fname, 'r')
	lines = file.readlines()

	width = 16
	height = 16
	size = width * height
	classes = 10

	images = [];
	labels = [];
	fnumber = 0;

	for line in lines:
		data = line.split(' ')
		image = [];
		label = [];

		for i in range(0, size):
			image.append(int(float(data[i])))
		images.append(image)
		
		for i in range(size, size + classes):
			label.append(int(float(data[i]))) 
		labels.append(label)

		fnumber += 1
		#if fnumber < 10:
			#image_utils.show(image, width, height)
		#image_utils.save('./dataset/semeion/images/' + str(fnumber) + '.png', array, width, height)

	#Shuffle data
	images_shuffle = []
	labels_shuffle = []
	indexes = list(range(len(images)))
	shuffle(indexes)
	for i in indexes:
		images_shuffle.append(images[i])
		labels_shuffle.append(labels[i])

	images = images_shuffle
	labels = labels_shuffle

	samples = len(lines)
	train_samples = 1300
	test_samples = 1100

	#Train set
	image_train = numpy.array(images[:train_samples], dtype=numpy.uint8)
	image_train = image_train.reshape(train_samples, width, height, 1)

	label_train = numpy.array(labels[:train_samples], dtype=numpy.uint8)

	train = dataset.DataSet(image_train, label_train, reshape=True)

	#test set
	image_test = numpy.array(images[test_samples:], dtype=numpy.uint8)
	image_test = image_test.reshape(samples - test_samples, width, height, 1)

	label_test = numpy.array(labels[test_samples:], dtype=numpy.uint8)

	test = dataset.DataSet(image_test, label_test, reshape=True)

	return Datasets(train=train, validation=data, test=test)
