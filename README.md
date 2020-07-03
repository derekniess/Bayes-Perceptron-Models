# Bayes-Perceptron-Models
This project explores two effective methods for AI-based image classification. Although I implemented these classification models for 
identifying the clothing types shown in images, the same models can easily be refactored to identify any other classification set, such as
digits, vehicles, animals, etc., provided one has a decent training/test dataset. For this project, I used the Fashion Mnist dataset of 50000
training examples and 10000 test examples.

## Model 1: Naive Bayes Model
### Features
Each image consists of 28*28 pixels which we represent as a flattened array of size 784, where each feature/pixel F<sub>i</sub> takes on 
intensity values from 0 to 255 (8 bit grayscale).

### Training
The goal of the training stage is to estimate the likelihoods **P(F<sub>i</sub> | class)** for every pixel location i and for every fashion item class (T-shirt, 
Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot, etc.). The likelihood estimate is defined as:

> **P(F<sub>i</sub> = f | class) = (# of times pixel i has value f in training examples from this class) / (Total # of training examples from this class)**

Smoothing the likelihoods was important to ensure there were no zero counts. More specifically, **Laplace smoothing** is a very simple method that 
increases the observation count of every value f by some constant k. This corresponds to adding k to the numerator above, and k*V to the denominator 
(where V is the number of possible values the feature can take on). The higher the value of k, the stronger the smoothing. I experimented with 
different values of k to find one that gave the highest classification accuracy, which happened to be a value of 1.

Priors P(class) were estimated by the empirical frequencies of different classes in the training set.

### Testing
Maximum a posteriori (MAP) classification was performed using the learned Bayes model. Suppose a test image has feature values f<sub>1</sub>, f<sub>2</sub>, ... , f<sub>784</sub>. 
According to this model, the posterior probability (up to scale) of each class given the digit is given by:

> **P(class) ⋅ P(f<sub>1</sub> | class) ⋅ P(f<sub>2</sub> | class) ⋅ ... ⋅ P(f<sub>784</sub> | class)**

However, in order to avoid underflow, we should work with the log of the above probabilities:

> **log(P(class)) ⋅ log(P(f<sub>1</sub> | class)) ⋅ log(P(f<sub>2</sub> | class)) ⋅ ... ⋅ log(P(f<sub>784</sub> | class))**


### Evaluation
Average Classification Rate: **74.21%**

#### Feature Likelihood Visualization Plots:
When using classifiers in real domains, it is important to be able to inspect what they have learned. One way to inspect a 
naive Bayes model is to look at the most likely features for a given label. Another tool for understanding the parameters is 
to visualize the feature likelihoods for high intensity pixels of each class. Here high intensities refer to pixel values from 
128 to 255. Therefore, the likelihood for high intensity pixel feature Fi of class c1 is sum of probabilities of the top 128 
intensities at pixel location i of class c1.

![feature visual](/likelihood_visual_bayes.png)

#### Test examples of each class that have the highest (H) and lowest (L) posterior probabilites:
![HL examples](/ex_classes_bayes.png)

#### Confusion Matrix:
![conf matrix](/conf_matrix_bayes.png)

TODO:
1) Resize images
2) Add probability tables
3) Add write-up for perceptron model
