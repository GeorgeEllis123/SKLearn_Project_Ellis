# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
# matplotlib 3.3.1
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import imutils

def getFlattened(imagepath):
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    dim = (64, 64)
    #updated = centerContour(img)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    flattened = np.reshape(resized, 4096)
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

trainX, testX, trainY,testY = train_test_split(images, labels, test_size = 0.3, shuffle = True)

classifier = LogisticRegression(max_iter = 20000)
#classifier = SGDClassifier(random_state=42, max_iter=20000, tol=1e-3)
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)

correct = 0
incorrect = 0
for pred, gt in zip(preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

plot_confusion_matrix(classifier, testX, testY)
plt.show()