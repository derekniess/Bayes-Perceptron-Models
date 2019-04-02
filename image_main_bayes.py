# Main function for train, test and visualize

import numpy as np
from perceptron import MultiClassPerceptron
from naive_bayes import NaiveBayes
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import math

def load_dataset(data_dir=''):
    # Load the train and test examples 
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    x_test = np.load("data/x_test.npy")
    y_test = np.load("data/y_test.npy")

    return x_train, y_train, x_test, y_test


def plot_visualization(images, classes, cmap):
    # Plot the visualizations
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax[i%2, i//2].imshow(images[:, i].reshape((28, 28)), cmap=cmap)
        ax[i%2, i//2].set_xticks([])
        ax[i%2, i//2].set_yticks([])
        ax[i%2, i//2].set_title(classes[i])
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix
    Normalization can be applied by setting `normalize=True`
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':

    # Load dataset 
    x_train, y_train, x_test, y_test = load_dataset()
    # Initialize naive bayes model
    num_class = len(np.unique(y_train))
    feature_dim = len(x_train[0]) 
    num_value = 256
    NB = NaiveBayes(num_class,feature_dim,num_value)
    # Train model
    NB.train(x_train,y_train)
    # Feature likelihood for high intensity pixels 
    feature_likelihoods = NB.intensity_feature_likelihoods(NB.likelihood)
    # Visualize the feature likelihoods for high intensity pixels 
    class_names = np.array(["T-shirt/top","Trouser","Pullover","Dress","Coat",
                            "Sandal","Shirt","Sneaker","Bag","Ankle boot"])
    plot_visualization(feature_likelihoods, class_names, "Greys")
    # Classify the test sets
    accuracy, y_pred = NB.test(x_test,y_test)

    # prints off probabilities of each example image that will be used for report
    # for i in range(NB.num_class):
    #     high_prob = NB.strongest_class_ex[i][0]
    #     high_img_num = NB.strongest_class_ex[i][1]
    #     low_prob = NB.weakest_class_ex[i][0]
    #     low_img_num = NB.weakest_class_ex[i][1]
    #     print("*****************************************")
    #     print("strongest example of class ", i, ": ", high_img_num)
    #     print("highest probability of class ", i, ": ", high_prob)
    #     print("weakest example of class ", i, ": ", low_img_num)
    #     print("lowest probability of class ", i, ": ", low_prob)
    # print("*****************************************")

    """ 
    Uses image indices collected from running the test set on the model. These indicees are used here to get
    the image pixel data of those images and stores them in mixed_imgs[] that will be used to print out images later.
    """
    mixed_imgs = np.zeros((num_class*2, 28, 28))
    for i in range(NB.num_class):
        h_image_idx = NB.strongest_class_ex[i][1]
        l_image_idx = NB.weakest_class_ex[i][1]
        for y_feature_idx in range(28):
            for x_feature_idx in range(28):
                mixed_imgs[i*2, y_feature_idx, x_feature_idx] = x_test[h_image_idx, y_feature_idx*28 + x_feature_idx]
                mixed_imgs[i*2+1, y_feature_idx, x_feature_idx] = x_test[l_image_idx, y_feature_idx*28 + x_feature_idx]

    classes_out = np.array(["T-shirt/top - H","T-shirt/top - L","Trouser - H","Trouser - L","Pullover - H","Pullover - L",
                            "Dress - H","Dress - L","Coat - H","Coat - L","Sandal - H","Sandal - L","Shirt - H","Shirt - L",
                            "Sneaker - H","Sneaker - L","Bag - H","Bag - L","Ankle boot - H","Ankle boot - L"])
    
    # prints out the example images of each class
    columns = 4
    rows = 5
    fig, ax = plt.subplots(rows, columns, figsize=(8,8))
    for i in range(rows):
        for j in range(columns):
            ax[i, j].xaxis.set_visible(False)
            ax[i, j].yaxis.set_visible(False)
            ax[i, j].set_title(classes_out[i*columns + j])
            ax[i, j].imshow(mixed_imgs[i*columns + j], cmap="gray")
    plt.show()
  
    # print("accuracy = ", accuracy)
    
    # Plot confusion matrix 
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
    plt.show() 

    
 
