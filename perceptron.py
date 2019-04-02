import numpy as np
import sys
import math

class MultiClassPerceptron(object):
	def __init__(self,num_class,feature_dim):
		"""
		Initialize a multi class perceptron model. 
		w is the set of weights for ea class to be applied to image features
			- the last index of feature_dim is assumed to be the bias term.
			- self.w[:,0] = [w1,w2,w3...,BIAS] get first column (class 0) where wi corresponds to each feature dimension

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		"""

		self.w = np.zeros((feature_dim+1,num_class))

		self.strongest_class_ex = np.zeros((num_class , feature_dim))
		self.weakest_class_ex = np.zeros((num_class , feature_dim))
		self.strong_val = np.full(num_class, -math.inf)
		self.weak_val = np.full(num_class, math.inf)


	def train(self,train_set,train_label):
		""" 
		Train perceptron model with training dataset. 

		Args:
		    train_set(numpy.ndarray): training examples with a dim of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dim of (# of examples)
		"""

		for time in range(1):
			img_idx = 0
			for curr_image in train_set:
				curr_class = 0
				set = 0
				max = -math.inf
				# add identity to compensate for bias in weights
				curr_image = np.append(curr_image, [1])
				# apply weights of a class to image features
				for curr_weights in self.w.T:				
					class_val = np.dot(curr_weights, curr_image)
					# we want to keep track of argmax of all classes
					if class_val > max:
						max = class_val
						set = curr_class
					curr_class = curr_class + 1
				# when guessed wrong, we want to update weights accordingly
				if set != train_label[img_idx]:
					self.w[:, train_label[img_idx]] = self.w[:, train_label[img_idx]] + 0.2*curr_image
					self.w[:, set] = self.w[:, set] - 0.2*curr_image
				img_idx = img_idx + 1


	def test(self,test_set,test_label):
		""" 
		Tests a trained naive bayes model on testing dataset, by performing maximum a posteriori (MAP) classification.  
		The accuracy is computed as the average of correctness by comparing between predicted label and true label. 

		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples)

		Returns:
			accuracy(float): average accuracy value  
			pred_label(numpy.ndarray): predicted labels with a dim of (# of examples)
		""" 
		accuracy = 0 
		pred_label = np.zeros((len(test_set)))
		wrong = 0
		img_idx = 0

		# on a per image basis, predict the image
		for curr_image in test_set:
			curr_class = 0
			set = 0
			max = -math.inf
			# add identity to compensate for bias in weights
			temp_image = np.append(curr_image, [1])
			# apply weights of a class to image features
			for curr_weights in self.w.T:				
				class_val = np.dot(curr_weights, temp_image)
				# we want to keep track of argmax of all classes
				if class_val > max:
					max = class_val
					set = curr_class
				curr_class = curr_class + 1
			pred_label[img_idx] = set
			# count how many times an image is incorrectly labelled
			if set != test_label[img_idx]:
				wrong = wrong + 1
			# keep track of extreme examples of ea class
			elif set == test_label[img_idx]:
				if self.strong_val[set] < class_val:
					self.strong_val[set] = class_val
					self.strongest_class_ex[set] = curr_image
				elif self.weak_val[set] > class_val:
					self.weak_val[set] = class_val
					self.weakest_class_ex[set] = curr_image
			img_idx = img_idx + 1
		accuracy = 1 - wrong / len(test_set)

		print ("accuracy is", accuracy)
		return accuracy, pred_label

	def save_model(self, weight_file):
		# Save the trained model parameters  
		np.save(weight_file,self.w)

	def load_model(self, weight_file):
		# Load the trained model parameters 
		self.w = np.load(weight_file)

