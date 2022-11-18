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

global imSize
imSize = 64

def displayGrid(images, preds, actual):
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

def centerContour(img):
    #blur = 5
    #blurred = cv2.GaussianBlur(img, (0, 0), 0)
    edge = cv2.Canny(img, 0, 100)
    cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cv2.drawContours(img, cnts, -1, (255, 0, 0), 10)
    cv2.imshow("sample", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def getFlattened(imagepath):
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    dim = (imSize, imSize)
    #updated = centerContour(img)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    flattened = np.reshape(resized, imSize*imSize)
    return flattened

def load_dataset(directory):
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

directory = "dataset"
labels, images = load_dataset(directory)

trainX, testX, trainY,testY = train_test_split(images, labels, test_size = 0.2, shuffle = True)

classifier = LogisticRegression(max_iter = 100)
classifier.fit(trainX, trainY)
print(classifier)
train_preds = classifier.predict(trainX)
test_preds = classifier.predict(testX)

correct = 0
incorrect = 0
for pred, gt in zip(test_preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print("Test Accuracy:")
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

correct = 0
incorrect = 0
for pred, gt in zip(train_preds, trainY):
    if pred == gt: correct += 1
    else: incorrect += 1
print("\nTrain Accuracy:")
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

displayGrid(testX, test_preds, testY)

plot_confusion_matrix(classifier, testX, testY)
plt.show()