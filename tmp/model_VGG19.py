import os                                 # Для работы с файлами
from PIL import Image, ImageEnhance                     # Отрисовка изображений
from tensorflow import keras
import random                             # Генерация случайных чисел 
import matplotlib.pyplot as plt           # Отрисовка графиков
import numpy as np                        # Библиотека работы с массивами
# Подключение нужных слоев из модуля tensorflow.keras.layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
# Подключение оптимизатора Adam
from tensorflow.keras.optimizers import Adam
import math                               # Математические функции
from keras.models import Model
from keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D

IMAGE_PATH = '/media/user/26C2BC47C2BC1CCD/D_projects/dipl_2/dipl/img/'

# Определение списка имен классов
CLASS_LIST = sorted(os.listdir(IMAGE_PATH))
# Определение количества классов
CLASS_COUNT = len(CLASS_LIST)
# Проверка результата
print(f'Количество классов: {CLASS_COUNT}, метки классов: {CLASS_LIST}')

i = 1
# Формирование пути к выборке одного класса диффекто
f'{IMAGE_PATH}{CLASS_LIST[i]}/'

#for cls in CLASS_LIST:
#    print(cls, ':', os.listdir(f'{IMAGE_PATH}{cls}/'))
#%matplotlib inline

# Создание заготовки для изображений всех классов
#fig, axs = plt.subplots(1, CLASS_COUNT, figsize=(25, 5))

# # Для всех номеров классов:
# for i in range(CLASS_COUNT):
#     # Формирование пути к папке содержимого класса
#     data_img_path = f'{IMAGE_PATH}{CLASS_LIST[i]}/' 
#     # Выбор случайного фото из i-го класса
#     img_path = data_img_path + random.choice(os.listdir(data_img_path)) 
#     # Отображение фотографии (подробнее будет объяснено далее)
#     axs[i].set_title(CLASS_LIST[i])
#     axs[i].imshow(Image.open(img_path))  
#     axs[i].axis('off')

# Отрисовка всего полотна
#plt.show()

data_files = []                           # Cписок путей к файлам картинок
data_labels = []                          # Список меток классов, соответствующих файлам
data_images = []                          # Пустой список для данных изображений


for class_label in range(CLASS_COUNT):    # Для всех классов по порядку номеров (их меток)
    class_name = CLASS_LIST[class_label]  # Выборка имени класса из списка имен
    class_path = IMAGE_PATH + class_name  # Формирование полного пути к папке с изображениями класса
    class_files = os.listdir(class_path)  # Получение списка имен файлов с изображениями текущего класса
    print(f'Размер класса {class_name} составляет {len(class_files)} фотографий')

    # Добавление к общему списку всех файлов класса с добавлением родительского пути
    data_files += [f'{class_path}/{file_name}' for file_name in class_files]

    # Добавление к общему списку меток текущего класса - их ровно столько, сколько файлов в классе
    data_labels += [class_label] * len(class_files)

print('Общий размер базы для обучения:', len(data_labels))

# Задание единых размеров изображений
IMG_WIDTH = 200                         # Ширина изображения
IMG_HEIGHT = 200                           # Высота изображения


for file_name in data_files:
    # Открытие и смена размера изображения
    img = Image.open(file_name).crop((0, 0, IMG_WIDTH, IMG_HEIGHT)) 
    img = np.array(img.convert('RGB'))
    img_np = np.array(img)                # Перевод в numpy-массив
    data_images.append(img_np)            # Добавление изображения в виде numpy-массива к общему списку

x_data = np.array(data_images)#, dtype=object)            # Перевод общего списка изображений в numpy-массив
y_data = np.array(data_labels)            # Перевод общего списка меток класса в numpy-массив

print(f'В массив собрано {len(data_images)} фотографий следующей формы: {img_np.shape}')
print(f'Общий массив данных изображений следующей формы: {x_data.shape}')
print(f'Общий массив меток классов следующей формы: {y_data.shape}')

# Нормированние массива изображений
x_data = x_data / 255

# Создание модели последовательной архитектуры
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    ])
# model = Sequential()

# model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(CLASS_COUNT, activation='softmax'))

model.summary()

# Подключение оптимизатора Adam
from tensorflow.keras.optimizers import Adam
# Компиляция модели
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Обучение модели сверточной нейронной сети подготовленных данных
store_learning = model.fit(x_data,  # ----------------- x_train, примеры набора данных
                           y_data,  # ----------------- y_train, метки примеров набора данных
                           validation_split=0.2,  # --- 0.2 - доля данных для валидационной (проверочной) выборки, 1-0.2=0.8 останется в обучающей
                           shuffle=True,  # ----------- перемешивание данных для равномерного обучения, соответствие экземпляра и метки сохраняется 
                           batch_size=25,  # ---------- размер пакета, который обрабатывает нейронка перед одним изменением весов
                           epochs=20,  # -------------- epochs - количество эпох обучения
                           verbose=1)  # -------------- 0 - не визуализировать ход обучения, 1 - визуализировать


#predIdxs = np.argmax(predIdxs, axis=1)

# Создание полотна для рисунка
plt.figure(1, figsize=(18, 5))

# Задание первой (левой) области для построения графиков
plt.subplot(1, 2, 1)
# Отрисовка графиков 'loss' и 'val_loss' из значений словаря store_learning.history
plt.plot(store_learning.history['loss'], 
         label='Значение ошибки на обучающем наборе')
plt.plot(store_learning.history['val_loss'], 
         label='Значение ошибки на проверочном наборе')
# Задание подписей осей 
plt.xlabel('Эпоха обучения')
plt.ylabel('Значение ошибки')
plt.legend()

# Задание второй (правой) области для построения графиков
plt.subplot(1, 2, 2) 
# Отрисовка графиков 'accuracy' и 'val_accuracy' из значений словаря store_learning.history
plt.plot(store_learning.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(store_learning.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
# Задание подписей осей 
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()

# Фиксация графиков и рисование всей картинки
plt.show()

#model.save("metal_detector.model", save_format="h5")


print("end")
