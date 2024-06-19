import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Rescaling
import time
#
# # Определение констант
# DIR = 'PetImages'  # Путь к директории с изображениями
# CATEGORIES = ['Cat', 'Dog']  # Категории изображений
# IMG_SIZE = 224  # Размер изображений (224x224 пикселя)
#
# # Функция для загрузки и предобработки изображений
# def load_data():
#     data = []
#     for category in CATEGORIES:
#         path = os.path.join(DIR, category)  # Путь к категории
#         label = CATEGORIES.index(category)  # Метка категории
#         for img in os.listdir(path):  # Перебор всех изображений в категории
#             try:
#                 img_path = os.path.join(path, img)  # Путь к изображению
#                 arr = cv2.imread(img_path)  # Загрузка изображения
#                 arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))  # Изменение размера изображения
#                 data.append([arr, label])  # Добавление изображения и его метки в данные
#                 print(img_path)
#             except Exception as e:
#                 print(f"Error loading image {img_path}: {e}")  # Обработка ошибок при загрузке изображения
#     return data
#
# # Загрузка и перемешивание данных
# data = load_data()
# random.shuffle(data)
#
# # Разделение данных на признаки (X) и метки (y)
# X, y = zip(*data)
# X = np.array(X)
# # X = np.array(X, dtype=np.float32)
#
# y = np.array(y)
#
# # Нормализация изображений
# X = X / 255.0
#
# # Сохранение данных с использованием pickle
# pickle.dump(X, open('X_animal2.pkl', 'wb'))
# pickle.dump(y, open('y_animal2.pkl', 'wb'))
#
# # Загрузка данных из pickle
# X = pickle.load(open('X_animal2.pkl', 'rb'))
# y = pickle.load(open('y_animal2.pkl', 'rb'))
#
# # Функция для создания модели
# def create_model(dense_layers, conv_layers, neurons):
#     model = Sequential()
#     model.add(Rescaling(1. / 255, input_shape=(IMG_SIZE, IMG_SIZE, 3)))  # Масштабирование входных данных
#
#     for _ in range(conv_layers):
#         model.add(Conv2D(neurons, (3, 3), activation='relu'))  # Сверточный слой
#         model.add(MaxPooling2D((2, 2)))  # Слой подвыборки
#
#     model.add(Flatten())  # Преобразование данных в одномерный массив
#     for _ in range(dense_layers):
#         model.add(Dense(neurons, activation='relu'))  # Полносвязный слой
#     model.add(Dense(len(CATEGORIES), activation='softmax'))  # Выходной слой
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])  # Компиляция модели
#     return model
#
# # Параметры для модели
# dense_layers = [1]
# conv_layers = [3]
# neurons = [64]
#
# # Создание и обучение модели
# model = create_model(dense_layers[0], conv_layers[0], neurons[0])
# history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
#
# # Функция для построения графиков истории обучения
# def plot_history(history):
#     # Построение графика точности
#     fig = plt.figure()
#     plt.plot(history.history['accuracy'], color='teal', label='accuracy')
#     plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
#     fig.suptitle('Accuracy', fontsize=20)
#     plt.legend(loc="upper left")
#     plt.show()
#
#     # Построение графика потерь
#     fig = plt.figure()
#     plt.plot(history.history['loss'], color='teal', label='loss')
#     plt.plot(history.history['val_loss'], color='orange', label='val_loss')
#     fig.suptitle('Loss', fontsize=20)
#     plt.legend(loc="upper left")
#     plt.show()
#
# plot_history(history)
#
# # Сохранение модели
# model.save('cat_dog_classifier.keras')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Rescaling


DIR = 'PetImages'
CATEGORIES = ['Cat', 'Dog', 'Rabbit']
IMG_SIZE = 224  # Размер изображений (224x224 пикселя)
#
# # Функция для загрузки и предобработки изображений
def load_data():
    data = []
    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        label = CATEGORIES.index(category)
        for img in os.listdir(path):  # Перебор всех изображений в категории
            try:
                img_path = os.path.join(path, img)  # Путь к изображению
                arr = cv2.imread(img_path)  # Загрузка изображения
                arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))
                data.append([arr, label])
                print(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")  # Обработка ошибок при загрузке изображения
    return data

# Загрузка и перемешивание данных
data = load_data()
random.shuffle(data)

X, y = zip(*data)
X = np.array(X)
y = np.array(y)

# Нормализация изображений
X = X / 255.0

pickle.dump(X, open('X_animal3.pkl', 'wb'))
pickle.dump(y, open('y_animal3.pkl', 'wb'))

# Загрузка данных из pickle
X = pickle.load(open('X_animal3.pkl', 'rb'))
y = pickle.load(open('y_animal3.pkl', 'rb'))


def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))  # Сверточный слой
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))  # Сверточный слой
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))  # Сверточный слой
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(CATEGORIES), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # Компиляция модели
    return model


# Создание и обучение модели
model = create_model()
history = model.fit(X, y, epochs=7, batch_size=32, verbose=1)

# Функция для построения графиков истории обучения

def plot_history(history):
    # Построение графика точности
    fig = plt.figure()
    plt.plot(history.history['accuracy'], color='teal', label='accuracy')
    # plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # Построение графика потерь
    fig = plt.figure()
    plt.plot(history.history['loss'], color='teal', label='loss')
    # plt.plot(history.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

plot_history(history)

# Сохранение модели
model.save('animal_classifier.keras')