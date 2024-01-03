import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import MaxPooling2D, Conv2D, Flatten,Activation,BatchNormalization
from keras.optimizers import Adam
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from tqdm import tqdm
img_array = []

label_data = np.load("label_data_basic.npy")

label_name = np.load("label_basic.npy")

data = np.load("data_basic.npy")

labelncoder = LabelEncoder()
label_data = labelncoder.fit_transform(label_data)
label_data = to_categorical(label_data, 52)

data = np.load("data_basic.npy")
data = np.array(data)
data = data/255

x_train, x_test, y_train, y_test = train_test_split(data, label_data, test_size=0.2)
models = Sequential()

# models.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(50,50,3),activation='relu'))
# models.add(MaxPooling2D(pool_size=(2,2)))
# models.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
# models.add(MaxPooling2D(pool_size=(2,2)))
# models.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
# models.add(MaxPooling2D(pool_size=(2,2)))
# models.add(Flatten())
# models.add(Dense(512,activation='relu'))
# models.add(Dense(52, activation='softmax'))

models.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3), padding='same'))
models.add(Activation("relu"))
models.add(BatchNormalization())
models.add(Conv2D(32, (3, 3), padding="same"))
models.add(Activation("relu"))
models.add(BatchNormalization())
models.add(MaxPooling2D(pool_size=(2,2)))
models.add(Conv2D(64, (3, 3), padding="same"))
models.add(Activation("relu"))
models.add(BatchNormalization())
models.add(Conv2D(64, (3, 3), padding="same"))
models.add(Activation("relu"))
models.add(BatchNormalization())
models.add(MaxPooling2D(pool_size=(2,2)))
models.add(Flatten())
models.add(Dense(512))
models.add(Activation("relu"))
models.add(BatchNormalization())
models.add(Dense(52))
models.add(Activation("softmax"))
opt = Adam(learning_rate=0.001)
models.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer= opt)
models.summary()
history = models.fit(x_train, y_train, validation_split= 0.1, epochs=3, batch_size=32)
train_loss, test_accuracy = models.evaluate(x_train, y_train)
print(f'train Accuracy: {test_accuracy * 100:.2f}%')
test_loss, test_accuracy = models.evaluate(x_test, y_test )
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

save_model = models.save("model_VGG.h5")

loss =history.history['loss']


acc =history.history['accuracy']


plt.plot(acc, label = 'Accuracy')

plt.plot(loss, label = 'loss')
plt.legend()
plt.show()