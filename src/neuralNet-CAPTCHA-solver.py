"""
Author: 
    Jake Scheetz
Date: 
    June 2022
Description: 
    Program looks at a Really Simple CAPTCHA (rsCAPTCHA) and then splits each character
    and places it into a supvised classifier placeholder for training. Classification is
    performed once for each character split from the CAPTCHA. Inspiration for the program
    taken from the "Machine Learning Cookbook for Cyber Security" publication.

    --> This particular script is used to train a neural network to solve rsCAPTCHA's

    --> Each script's main logic is placed at the bottom of the code for clarity

Dependencies: 
    - opencv-python
    - imutils
    - numpy
    - sklearn
    - keras
    - tensorflow
"""

# imports
import cv2
import imutils
import os
from imutils import paths
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from imagePreprocessing import *
# --------------

# global vars
captcha_processing_output_folder = "extracted-letter-images"

images = []
labels = []

num_classes = 32
# --------------

# ~~~~~~~~~~~~~~~~~~ functions ~~~~~~~~~~~~~~~~~~
def resizeImage(image, newHeight, newWidth):
    """Resizes a given image to the specified dimensions"""
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=newWidth)
    else:
        image = imutils.resize(image, height=newHeight)
    padWidth = int((newWidth - image.shape[1]) / 2.0)
    padHeight = int((newHeight - image.shape[0]) / 2.0)
    imageBorder = cv2.copyMakeBorder(image, padHeight, padHeight, padWidth, padWidth, cv2.BORDER_REPLICATE)
    imageWithBorderResized = cv2.resize(imageBorder, (newWidth, newHeight))
    return imageWithBorderResized

def readImage(pathToImage):
    """Read in an image"""
    image = cv2.imread(pathToImage)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resizeImage(image, 20, 20)
    image = np.expand_dims(image, axis=2)
    return image
# ~~~~~~~~~~~~~~~~~~ end functions ~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~ Main Logic ~~~~~~~~~~~~~~~~~~
# read in each image of a letter and record the label
for pathToImage in imutils.paths.list_images(captcha_processing_output_folder):
    imageFile = readImage(pathToImage)
    label = pathToImage.split(os.path.sep)[-2]
    images.append(imageFile)
    labels.append(label)

# normalize all images:
#   - rescale pixels 0-1
#   - convert labels to numpy array
images = np.array(images, dtype="float") / 255.0
labels = np.array(labels)

# creating a train-test split
(x_train, x_test, y_train, y_test) = train_test_split(
    images, labels, test_size=0.3, random_state=11
)

# encode labels
label_binarizer = LabelBinarizer().fit(y_train)
y_train = label_binarizer.transform(y_train)
y_test = label_binarizer.transform(y_test)


# Defining the Neural Network
NN_model = Sequential()
NN_model.add(
    Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu")
)
NN_model.add(
    MaxPooling2D(pool_size=(2,2), strides=(2,2))
)
NN_model.add(
    Conv2D(50, (5,5), padding="same", activation="relu")
)
NN_model.add(
    MaxPooling2D(pool_size=(2,2), strides=(2,2))
)
NN_model.add(Flatten())
NN_model.add(Dense(512, activation="relu"))
NN_model.add(Dense(num_classes, activation="softmax"))
NN_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
NN_model.summary()

# fitting the defined NeuralNet to the training (preprocessed) data
NN_model.fit(
    x_train,
    y_train,
    validation_data=(x_train, y_train),
    batch_size=16,
    epochs=5,
    verbose=1,
)

# hard coding a captcha image to break for debugging
# ideally this will be converted to logic to solve the observed captcha
captcha = "captcha-images/NZH2.png"







