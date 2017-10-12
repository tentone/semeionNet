import gzip
import collections
import numpy
import dataset

from tensorflow.python.framework import dtypes

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#Auxiliar function to read integer from bytestream
def read32(bytestream):
	dt = numpy.dtype(numpy.uint32).newbyteorder('>')
	return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

#Extract the images into a 4D uint8 numpy array [index, y, x, depth]
#Receives a file object that can be passed into a gzip reader
def read_images(f):
	with gzip.GzipFile(fileobj=f) as bytestream:
		magic = read32(bytestream)
		if magic != 2051:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
		images = read32(bytestream)
		rows = read32(bytestream)
		cols = read32(bytestream)
		buf = bytestream.read(rows * cols * images)
		data = numpy.frombuffer(buf, dtype=numpy.uint8)
		data = data.reshape(images, rows, cols, 1)
		return data

#Convert class labels from scalars binary class vector
def value_to_class_vector(value, classes):
	labels = value.shape[0]
	offset = numpy.arange(labels) * classes
	vector = numpy.zeros((labels, classes))
	vector.flat[offset + value.ravel()] = 1
	return vector


#Extract the labels into a 1D uint8 numpy array [index]
#Receives a file object that can be passed into a gzip reader
def read_labels(f, classes = 10):
	with gzip.GzipFile(fileobj=f) as bytestream:
		magic = read32(bytestream)
		if magic != 2049:
			raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
		buf = bytestream.read(read32(bytestream))
		
		labels = numpy.frombuffer(buf, dtype=numpy.uint8)

		return value_to_class_vector(labels, classes)

#Read mnist data sets
def read_data_mnist(path='./dataset/mnist/', dtype=dtypes.float32, reshape=True, validation_size=5000, seed=None):
	file = open(path + TRAIN_IMAGES, 'rb')
	train_images = read_images(file)

	file = open(path + TRAIN_LABELS, 'rb')
	train_labels = read_labels(file)

	file = open(path + TEST_IMAGES, 'rb')
	test_images = read_images(file)

	file = open(path + TEST_LABELS, 'rb')
	test_labels = read_labels(file)

	if not 0 <= validation_size <= len(train_images):
		raise ValueError('Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size))

	validation_images = train_images[:validation_size]
	validation_labels = train_labels[:validation_size]
	train_images = train_images[validation_size:]
	train_labels = train_labels[validation_size:]

	train = dataset.DataSet(train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
	validation = dataset.DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape, seed=seed)
	test = dataset.DataSet(test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

	return Datasets(train=train, validation=validation, test=test)
