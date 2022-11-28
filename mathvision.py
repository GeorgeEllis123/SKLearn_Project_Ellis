# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
# matplotlib 3.3.1
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import numpy as np
import os
import imutils

# The dimensions the images will be scaled to
global imSize
imSize = 64

def displayGrid(images, preds, actual):
    """ Displays a grid of several images with predicted and actual labels """
    fig = plt.figure(figsize=(8.,3.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1,4),
                     axes_pad=0.1,)
    img1 = np.reshape(images[0], (imSize,imSize))
    img2 = np.reshape(images[1], (imSize,imSize))
    img3 = np.reshape(images[2], (imSize,imSize))
    img4 = np.reshape(images[3], (imSize,imSize))
    i = 0
    for ax, im in zip(grid, [img1, img2, img3, img4]):
        label = "Prediction: " + str(preds[i]) + " Actual: " + str(actual[i])
        ax.set_xlabel(label, fontsize=7)
        ax.imshow(im)
        i += 1

def getFlattened(imagepath):
    """ Flattens and grayscales the image """
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    dim = (imSize, imSize)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    flattened = np.reshape(resized, imSize*imSize)
    return flattened

def load_dataset(directory):
    """ Loops through all of the folders and returns the images with labels 
    based on the folder they are in """
    data = {"labels": [],
            "images": []}

    # Loops through all of the folders in the dataset
    for folder in os.listdir(directory):
        f = os.path.join(directory, folder) # Gets the folder path

        for filename in os.listdir(f):  # Loops through all the files in current folder
            file = os.path.join(f, filename) # Gets the file path
            image = getFlattened(str(file))  # Flattens the image
            data["labels"].append(folder)   # Adding the labels based on the folder
            data["images"].append(image)    # Adding the flattened image

    return data["labels"], data["images"]

def main():
    # Gets the dataset
    directory = "dataset"
    labels, images = load_dataset(directory)

    # Splits the dataset
    trainX, testX, trainY,testY = train_test_split(images, labels, test_size = 0.2, shuffle = True)

    # Defines and fits the classifier
    classifier = LogisticRegression(max_iter = 100)
    classifier.fit(trainX, trainY)

    # Gets the test and train predictions
    train_preds = classifier.predict(trainX)
    test_preds = classifier.predict(testX)

    # Displays the train accuracy
    correct = 0
    incorrect = 0
    for pred, gt in zip(train_preds, trainY):
        if pred == gt: correct += 1
        else: incorrect += 1
    print("\nTrain Accuracy:")
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

    # Displays the test accuracy
    correct = 0
    incorrect = 0
    for pred, gt in zip(test_preds, testY):
        if pred == gt: correct += 1
        else: incorrect += 1
    print("\nTest Accuracy:")
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

    # Displays several images with predicted and actual labels
    displayGrid(testX, test_preds, testY)

    # Plots the confusion matrix
    plot_confusion_matrix(classifier, testX, testY)
    plt.show()

main()
