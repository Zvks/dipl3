import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import math
import os
import itertools
#from keras.utils 
import np_utils
import random
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
import json
from IPython.display import display
from PIL import Image
# We just have the file names in the x set. Let's load the images and convert them into array.
from keras.preprocessing.image import array_to_img, img_to_array, load_img

defect_class = []
with open('result/GOST_list_defect.json', 'r') as f:
    fl = json.load(f)
    for i in fl['GOST_list_defect']:
        print(i)
        defect_class.append(i)
f.close()
finish_list = []
short_finish_list = ''
tec_list = []

def get_img(image_file: str):
    IMAGE_PATH = image_file
    data = []
    # Определение списка имен классов
    IMG_LIST = sorted(os.listdir(IMAGE_PATH))
    print(IMG_LIST)
    for file_name in IMG_LIST:
        print(file_name)
        path = str(IMAGE_PATH + '/' + file_name)
        print(path)
        data.append(path)
    return data

def get_num(data_img):
    files = data_img
    images_as_array = []
    for file in files:
        images_as_array.append(img_to_array(load_img(file)))
    x_test = np.array(images_as_array)
    print('Test set shape : ',x_test.shape)
    x_test = x_test.astype('float32')/255
    return x_test

try:
    # Загрузка обученной модели из файла
    path_to_img = "result/app_img"
    path_to_model = 'result/16_model_4.h5'
    model = keras.models.load_model(path_to_model)
    print("Модель загружена")
    data_img = get_img(path_to_img)
    x_test = get_num(data_img)
    defect = model.predict(x_test)
    for i in defect:
        #print(i)
        n = 0
        for j in i:
            ans = defect_class[n][0] + ' = ' + str(j)
            #print(ans)
            finish_list.append(ans)        
            if j == max(i):
                short_finish_list += str(ans + ' ' + defect_class[n][1] +'\n')
                n += 1
    print(short_finish_list)      
except Exception:
    print('Ошибка')
else:
    print('Всё хорошо.')
finally:
    print('Завершение')

