import numpy

from tensorflow.python.framework import dtypes, random_seed

#DataSet constructor, used to store an monochromatic image based dataset
#dtype can be either uint8 to leave the input as [0, 255], or float32 to rescale into [0, 1].
#Seed arg provides for convenient deterministic testing.
class DataSet(object):
	def __init__(self, images, labels, dtype = dtypes.float32, reshape = True, seed = None):
		seed1, seed2 = random_seed.get_seed(seed)

		#If op level seed is not set, use whatever graph level seed is returned
		numpy.random.seed(seed1 if seed is None else seed2)
		dtype = dtypes.as_dtype(dtype).base_dtype

		if dtype not in (dtypes.uint8, dtypes.float32):
			raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

		assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
		
		self.num_examples = images.shape[0]

		#Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns](assuming depth == 1)
		if reshape:
			assert images.shape[3] == 1
			images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

		#Convert from [0, 255] -> [0.0, 1.0].
		if dtype == dtypes.float32:
			images = images.astype(numpy.float32)
			images = numpy.multiply(images, 1.0 / 255.0)

		self.images = images
		self.labels = labels
		self.epochs_completed = 0
		self.index_in_epoch = 0

	#Return the next batch_size examples from this data set
	def next_batch(self, batch_size, shuffle=True):
		start = self.index_in_epoch

		#Shuffle for the first epoch
		if self.epochs_completed == 0 and start == 0 and shuffle:
			perm0 = numpy.arange(self.num_examples)
			numpy.random.shuffle(perm0)
			self.images = self.images[perm0]
			self.labels = self.labels[perm0]
		
		#Get Next epoch
		if start + batch_size > self.num_examples:
			#Finished epoch
			self.epochs_completed += 1
			
			#Get the rest examples in this epoch
			rest_num_examples = self.num_examples - start
			images_rest_part = self.images[start:self.num_examples]
			labels_rest_part = self.labels[start:self.num_examples]

			#Shuffle the data
			if shuffle:
				perm = numpy.arange(self.num_examples)
				numpy.random.shuffle(perm)
				self.images = self.images[perm]
				self.labels = self.labels[perm]
			
			#Start next epoch
			start = 0
			self.index_in_epoch = batch_size - rest_num_examples
			end = self.index_in_epoch
			images_new_part = self.images[start:end]
			labels_new_part = self.labels[start:end]
			return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			return self.images[start:end], self.labels[start:end]
