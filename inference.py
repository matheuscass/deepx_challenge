# This Python file uses the following encoding: utf-8


# imports used
import numpy

from keras.models import load_model

from os import listdir
from os.path import isfile, join
import cv2

# loading model trained by train.ipynb
model = load_model('my_model.h5')

# Path to the images to be read
mypath='./read_images/'

# Reading images from mypath
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]),0 )
  image = cv2.resize(images[n], (28, 28))                 # Resize them to 28x28
  image = image.reshape(1,1,28,28).astype('float32')      # Reshapes the image to fit the network model
  image = (255-image)/255                                 # As the dataset used is composed of white
# digits against dark background, we have to use a "negative version" of the pixels
  pred = model.predict_classes(image)                     # Makes the prediction based on model
  pred_proba = model.predict_proba(image)                 # Gets the probability that the prediction is correct
  pred_proba = "%.2f%%" % (pred_proba[0][pred]*100)
  print(pred[0], " with confidence of ", pred_proba)
