import numpy
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

matplotlib.interactive(True)

def show(data, width, height):
	image = numpy.array(data).reshape(width, height)
	matplotlib.pyplot.imshow(image, cmap=matplotlib.cm.gray, interpolation="nearest")
	input('Press any key to continue!')

def save(fname, data, width, height):
	image = numpy.array(data).reshape(width, height)
	matplotlib.pyplot.imsave(fname, image, cmap=matplotlib.cm.gray)

def view_weights(weights, width, height):
	cdict = {
		'red':
		[
			(0.0,  1.0, 1.0),
			(0.25,  1.0, 1.0),
			(0.5,  0.0, 0.0),
			(1.0,  0.0, 0.0)
		],
		'green':
		[
			(0.0,  0.0, 0.0),
			(0.5,  0.0, 0.0),
			(0.75, 1.0, 1.0),
			(1.0,  1.0, 1.0)
		],
		'blue':
		[
			(0.0,  0.0, 0.0),
			(1.0,  0.0, 0.0)
		]
	}

	red_green = matplotlib.colors.LinearSegmentedColormap('green_red', cdict, 256)

	for i in range(0, 10):
		img = weights.flatten()[i::10].reshape((width, height))
		pyplot.imshow(img, cmap = red_green, clim=(-1, 1))
		if i == 0:
			pyplot.colorbar()
		input('Class ' + str(i))