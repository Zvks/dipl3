import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import math
import os
import itertools
#from keras.utils 
import np_utils
from sklearn.preprocessing import Normalizer, scale
from sklearn.datasets import load_files
import tensorflow as tf
from tensorflow import keras
from IPython.display import SVG
from keras.utils import model_to_dot
from keras.utils import plot_model
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
import IPython
from IPython.display import display
from PIL import Image
# We just have the file names in the x set. Let's load the images and convert them into array.
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import json
#%matplotlib inline
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)

# train_dir = '/media/user/26C2BC47C2BC1CCD/D_projects/dipl_2/dipl/img3/NEU_Metal_Surface_Defects_Data/train'
# val_dir = '/media/user/26C2BC47C2BC1CCD/D_projects/dipl_2/dipl/img3/NEU_Metal_Surface_Defects_Data/valid'
# test_dir='/media/user/26C2BC47C2BC1CCD/D_projects/dipl_2/dipl/img3/NEU_Metal_Surface_Defects_Data/test'
# print("Path Direcorty:      ",os.listdir("/media/user/26C2BC47C2BC1CCD/D_projects/dipl_2/dipl/img3/NEU_Metal_Surface_Defects_Data"))
# print("Train Direcorty:     ",os.listdir("/media/user/26C2BC47C2BC1CCD/D_projects/dipl_2/dipl/img3/NEU_Metal_Surface_Defects_Data/train"))
# print("Test Direcorty:      ",os.listdir("/media/user/26C2BC47C2BC1CCD/D_projects/dipl_2/dipl/img3/NEU_Metal_Surface_Defects_Data/test"))
# print("Validation Direcorty:",os.listdir("/media/user/26C2BC47C2BC1CCD/D_projects/dipl_2/dipl/img3/NEU_Metal_Surface_Defects_Data/valid"))

# # Distribution for 'Inclusion' surface defect
# print("Training Inclusion data:  ",len(os.listdir(train_dir+'/'+'Inclusion')))
# print("Testing Inclusion data:   ",len(os.listdir(test_dir+'/'+'Inclusion')))
# print("Validation Inclusion data:",len(os.listdir(val_dir+'/'+'Inclusion')))

train_dir = '/media/user/26C2BC47C2BC1CCD/D_projects/prct/dipl2/img2/NEU_Metal_Surface_Defects_Data/train'
val_dir = '/media/user/26C2BC47C2BC1CCD/D_projects/prct/dipl2/img2/NEU_Metal_Surface_Defects_Data/val'
test_dir='/media/user/26C2BC47C2BC1CCD/D_projects/prct/dipl2/img2/NEU_Metal_Surface_Defects_Data/test'
result_dir = '/media/user/26C2BC47C2BC1CCD/D_projects/prct/dipl2/result/'
print("Path Direcorty:      ",os.listdir("/media/user/26C2BC47C2BC1CCD/D_projects/prct/dipl2/img2/NEU_Metal_Surface_Defects_Data"))
print("Train Direcorty:     ",os.listdir("/media/user/26C2BC47C2BC1CCD/D_projects/prct/dipl2/img2/NEU_Metal_Surface_Defects_Data/train"))
print("Test Direcorty:      ",os.listdir("/media/user/26C2BC47C2BC1CCD/D_projects/prct/dipl2/img2/NEU_Metal_Surface_Defects_Data/test"))
print("Validation Direcorty:",os.listdir("/media/user/26C2BC47C2BC1CCD/D_projects/prct/dipl2/img2/NEU_Metal_Surface_Defects_Data/val"))

def result_dict(defect_class_dsc_path, result_path):
    try:
        defect_class_dsc = []
        list_defect_class_dsc = []
        class_list = sorted(os.listdir(defect_class_dsc_path))  

        for class_name in class_list:       
            defect_class_dsc.append(class_name)
            info = os.getxattr(defect_class_dsc_path + '/' + class_name,'user.description' ,follow_symlinks=True)
            info = info.decode("utf-8")
            defect_class_dsc.append(info)
            list_defect_class_dsc.append(defect_class_dsc)
            defect_class_dsc = []

        to_json = {
            "GOST_list_defect": list_defect_class_dsc
        }
        with open(result_path + 'GOST_list_defect.json', 'w') as f:
            f.write(json.dumps(to_json, sort_keys=False, indent=4, ensure_ascii=False, separators=(',', ': ')))
        f.close()
    except Exception:
        print('Ошибка')
    else:
        print(list_defect_class_dsc)
    finally:
        print('Завершение')
    return Exception
# Distribution for 'Inclusion' surface defect
# print("Training gost 21014_2022 non metallic inclusion training data:  ",len(os.listdir(train_dir+'/'+'gost_21014_2022_non_metallic_inclusion')))
# print("Testing gost 21014_2022 non metallic inclusion testing data:   ",len(os.listdir(test_dir+'/'+'gost_21014_2022_non_metallic_inclusion_test')))
# print("Validation gost 21014_2022 non metallic inclusion validation data:   ",len(os.listdir(val_dir+'/'+'gost_21014_2022_non_metallic_inclusion_val')))

# подготовка изображений для обучающей выборки
train_datagen = ImageDataGenerator(
    rescale = 1. / 255, # изменение масштаба(уменьшение в 255 раз)
    rotation_range = 8, # произвольное вращение изображения в диапазоне (от 0 до 8 в градусах)
    zoom_range = 0.1, # произвольное увелечение изображения на 10% 
    shear_range = 0.3, # угол сдвига против часовой стрелки в градусах 
    width_shift_range = 0.08, # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.08, # randomly shift images vertically (fraction of total height)
    vertical_flip = True, # randomly flip images
    horizontal_flip = True) # randomly flip images

# подготовка изображений для тестовой выборки
test_datagen = ImageDataGenerator(rescale = 1. / 255)

# Поток обучающих изображений партиями по 10 с использованием генератора train_datagen
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (200, 200),
        batch_size = 10,
        class_mode = 'categorical')

# Поток проверочных изображений партиями по 10 с использованием генератора test_datagen
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



# Построение сети 2D ConvNet(потому что изображение ч/б)

model = Sequential() # Sequential Keras API which is a linear stack of layers

model.add(Conv2D(filters = 32, # Количество фильтров (ядер), используемых с этим слоем
                 
                 kernel_size = (5, 5), # Размерность свертки 5x5
                 
                 activation = "relu", # Активационная функция - Rectified Linear Unit (ReLU)
                 
                 strides = 1, # Величена смещения окна (карты объектов) по каждому из измерений
                 
                 padding = "same", # Когда шаг = 1, выходная пространственная форма совпадает с входной пространственной формой
                 
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
        batch_size = 32,
        epochs = 100,
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

# First, we are going to load the file names and their respective target labels into numpy array! 



def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels
    
x_test, y_test,target_labels = load_dataset(test_dir)

no_of_classes = len(np.unique(y_test))
# y_test = np_utils.to_categorical(y_test, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)




def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)

x_test = x_test.astype('float32')/255

# Let's visualize test prediction.

y_pred = model.predict(x_test)

# plot a raandom sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(16, 9))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
model.save('result/16_model_6', save_format='h5')
result_dict(test_dir,result_dir)