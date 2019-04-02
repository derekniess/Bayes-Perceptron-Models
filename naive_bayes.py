import numpy as np
import math

class NaiveBayes(object):
	def __init__(self,num_class,feature_dim,num_value):
		"""
		Initialize a naive bayes model. 
		prior is Prob(class) with a dimension of (# of class)
			- estimates the empirical frequencies of different classes in the training set.
		
		likelihood is Prob(F_i = f | class) with a dimension of (# of features per image, # of possible values per pixel, # of class),
			- computes the probability of every pixel location i being value f for every class label.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		    num_value(int): number of possible values for each pixel 
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim

		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))

		# <min/max>_class_ex[i][0] returns the probability of the image of the ith class with the highest probability in the ith class
    	# <min/max>_class_ex[i][1] returns the image number of the ith class with the highest probability in the ith class
		self.strongest_class_ex = [[-math.inf, 0] for i in range(num_class)]
		self.weakest_class_ex = [[math.inf, 0] for i in range(num_class)]

	def train(self,train_set,train_label):
		""" 
		Train naive bayes model with training dataset. 

		Args:
		    train_set(numpy.ndarray): dim of (# of examples, feature_dim)
		    train_label(numpy.ndarray): dime of (# of examples)
		"""

		num_images = len(train_set)
		k = 1
		images_of_class = np.zeros((self.num_class))

		# count # images of each class
		for i, class_num in np.ndenumerate(train_label):
			images_of_class[class_num] += 1
			
		# calculate priors of each class
		for i in range(self.num_class):
			self.prior[i] = images_of_class[i] / num_images
			if self.prior[i] != 0.0:
				self.prior[i] = math.log(self.prior[i])	
		
		# Version 1: get counts for likelihoods
		# for image_num in range(num_images):
		# 	class_num = train_label[image_num]
		# 	for feature_num in range(self.feature_dim):
		# 		value_num = train_set[image_num, feature_num]
		# 		self.likelihood[feature_num, value_num, class_num] += 1

		# Version 2: get counts for likelihoods
		for (image_num, feature_num), value_num in np.ndenumerate(train_set):
			class_num = train_label[image_num]
			self.likelihood[feature_num, value_num, class_num] += 1

		# calculate likelihoods using above counts
		for (feature_num,value_num,class_num), idx_sum in np.ndenumerate(self.likelihood):
			likelihood_temp = (idx_sum + k) / (images_of_class[class_num] + k)
			if likelihood_temp != 0.0:
				likelihood_temp = math.log(likelihood_temp)
			self.likelihood[feature_num,value_num,class_num] = likelihood_temp

		pass

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
		class_acc = np.zeros(self.num_class)
		class_total = np.zeros(self.num_class)

		accuracy = 0
		pred_label = np.zeros((len(test_set)))

		# on a per image basis, predict the image
		for image_num in range(len(test_set)):
			MAP = -math.inf
			# will hold posterior probabilities of each class, init as priors of classes
			posterior_sum = np.array(self.prior, copy=True)

			# each iteration calculates a posterior of a class
			for class_num in range(self.num_class):

				# sum up the likelihoods: P(f1 to f784 | class = ci)
				for feature_idx in range(self.feature_dim):
					feature_val = test_set[image_num, feature_idx]
					f_likelihood = self.likelihood[feature_idx, feature_val, class_num]
					posterior_sum[class_num] += f_likelihood

				# update max posterior if better posterior found
				# MAP holds the current best guess's posterior probability of the image
				if posterior_sum[class_num] > MAP:
					MAP = posterior_sum[class_num]
					pred_label[image_num] = class_num

				# keep track of extreme examples of ea class
				if posterior_sum[class_num] > self.strongest_class_ex[class_num][0] and test_label[image_num] == class_num:
					self.strongest_class_ex[class_num][0] = posterior_sum[class_num]
					self.strongest_class_ex[class_num][1] = image_num
				if posterior_sum[class_num] < self.weakest_class_ex[class_num][0] and test_label[image_num] == class_num:
					self.weakest_class_ex[class_num][0] = posterior_sum[class_num]
					self.weakest_class_ex[class_num][1] = image_num

			# when we correctly predict an image
			if pred_label[image_num] == test_label[image_num]:
				accuracy += 1
				class_acc[test_label[image_num]] += 1

			# need to count # of instances of each class to calc class avg
			class_total[test_label[image_num]] += 1

		accuracy = accuracy / len(test_set)
		
		pass
		return accuracy, pred_label


	def save_model(self, prior, likelihood):
		# Save the trained model parameters  
		np.save(prior, self.prior)
		np.save(likelihood, self.likelihood)

	def load_model(self, prior, likelihood):
		# Load the trained model parameters
		self.prior = np.load(prior)
		self.likelihood = np.load(likelihood)

	def intensity_feature_likelihoods(self, likelihood):
	    """
	    Helps generate visualization of trained likelihood images.
	    It gets the feature likelihoods for high intensity pixels for each of the classes
	    by summing the probabilities of the top 128 intensities at each pixel location:
	    	- sum k<-128:255 P(F_i = k | c)

	    Returns:
	        feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dim of
	            								(# of features/pixels per image, # of class)
	    """
	    
	    feature_likelihoods = np.zeros((likelihood.shape[0],likelihood.shape[2]))

	    for class_num in range(likelihood.shape[2]):
	    	for feature_num in range(likelihood.shape[0]):
	    		for value in range(128, 256):
	    			feature_likelihoods[feature_num, class_num] = feature_likelihoods[feature_num, class_num] + likelihood[feature_num, value, class_num]

	    return feature_likelihoods



