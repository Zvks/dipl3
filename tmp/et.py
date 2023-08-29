import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import math
import os
import itertools
from keras.utils import np_utils
from sklearn.preprocessing import Normalizer, scale
from sklearn.datasets import load_files
import tensorflow as tf
from tensorflow import keras
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import missingno as msno
%matplotlib inline
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)

train_dir = '/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/train'
val_dir = '/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/valid'
test_dir='/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/test'
print("Path Direcorty:      ",os.listdir("/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data"))
print("Train Direcorty:     ",os.listdir("/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/train"))
print("Test Direcorty:      ",os.listdir("/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/test"))
print("Validation Direcorty:",os.listdir("/kaggle/input/neu-metal-surface-defects-data/NEU Metal Surface Defects Data/valid"))

train_datagen = ImageDataGenerator(
    rescale = 1. / 255, # rescaling
    rotation_range = 8, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image 
    shear_range = 0.3, # shear angle in counter-clockwise direction in degrees  
    width_shift_range = 0.08, # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.08, # randomly shift images vertically (fraction of total height)
    vertical_flip = True, # randomly flip images
    horizontal_flip = True) # randomly flip images


test_datagen = ImageDataGenerator(rescale = 1. / 255)

# Flow training images in batches of 10 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (200, 200),
        batch_size = 10,
        class_mode = 'categorical')

# Flow validation images in batches of 10 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size = (200, 200),
        batch_size = 10,
        class_mode = 'categorical')

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98 ): # Stop training the model at 98% traning accuracy
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True

model = Sequential() # Sequential Keras API which is a linear stack of layers

model.add(Conv2D(filters = 32, # The number of filters (Kernels) used with this layer
                 
                 kernel_size = (5, 5), # The dimensions of the feature map
                 
                 activation = "relu", # Activation function - Rectified Linear Unit (ReLU)
                 
                 strides = 1, # How much the window (feature map) shifts by in each of the dimensions
                 
                 padding = "same", # When stride = 1, output spatial shape is the same as input spatial shape
                 
                 # There are two conventions for shapes of images tensors: the channels-last convention 
                 # (used by TensorFlow) and the channels-first convention (used by Theano)." 
                 # Deep Learning with Python - François Chollet
                 data_format = "channels_last",
                 
                 input_shape = (200, 200, 3))) # Input image dimensions

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", 
                 strides = 1, padding = "same", data_format = "channels_last"))

# Max Pooling reduces the spatial dimensions of the feature maps before the fully connected layers
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu", 
                 strides = 1, padding = "same", data_format = "channels_last"))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu", 
                 strides = 1, padding = "same", data_format = "channels_last"))

model.add(MaxPooling2D(pool_size = (2, 2)))
    
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = "relu", 
                 strides = 1, padding = "same", data_format = "channels_last"))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = "relu", 
                 strides = 1, padding = "same", data_format = "channels_last"))

model.add(MaxPooling2D(pool_size = (2, 2)))

# To help avoid overfitting we can add Dropout. 
# This randomly drops some percentage of neurons, and thus the weights become re-aligned
model.add(Dropout(0.1))

# Finally, we can add a flatten layer to map the input to a 1D vector
# We then add fully connected (dense) layers after some convolutional/pooling layers.

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(6, activation = "softmax")) # activation function for Multi-Class Classification

optimizer = Adam(lr = 0.00002)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
              
model.summary()

callbacks = myCallback()
history = model.fit(train_generator,
        batch_size = 64,
        epochs = 40,
        validation_data = validation_generator,
        callbacks = [callbacks],
        verbose = 1, shuffle = True)

plt.figure(figsize = (12, 6))
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label = "Training accuracy")
plt.plot(epochs, val_accuracy, 'r', label = "Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure(figsize = (12, 6))
plt.plot(epochs, loss, 'b', label = "Training loss")
plt.plot(epochs, val_loss, 'r', label = "Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()