# SemeionNet
 - A set of machine learning experiments with the semeion and MNIST handwritten digit dataset using  tensorflow
 - The objective of this experiment was to test multiple classification methods using the semeion handwriting dataset and measure performance of different classifiers implementation in tensorflow. 
 - The code available was tested with Python 3.5 and Tensorflow 1.1

## Dataset
 - The semeion dataset is composed of 1593 handwritten digits from 80 persons that were scanned and stretched to a 16x16 size image.
 	- http://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit
 - MNIST dataset is a subset of the NIST dataset that has over 60000 handwritten digits.
 	- http://yann.lecun.com/exdb/mnist/

## How to run
 - 

## Install
 - Before running the examples in the repository, install the dependencies indicated bellow
	 - tensorflow
	 - matplotlib
	 - sklearn
	 - pandas
	 - numpy

## Results
 - The results bellow were obtained, using 1300 random entries from the semeio dataset to train the classifier and 400 random entries to test the trained model.
 - The results obtained are expected, except for the recurrent neural net that developed patterns to predict the python internal pseudo-random number generator.
 - Tests were run on a Core i5 6500 CPU with 24GB of RAM.

| Classifier  | Time | Accuracy |
| ----------- | ---- | -------- |
| Softmax     | 40.8 | 94.32%   |
| KNN         | 3.85 | 94.52%   |
| Perceptron  | 11.4 | 97.16%   |
| CNN         | 89.7 | 96.95%   |
| RNN         | 35.1 | 97.56%   |
