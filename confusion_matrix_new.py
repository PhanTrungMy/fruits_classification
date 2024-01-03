import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
import seaborn as sns
from keras.layers import Dense,Dropout
from keras.layers import MaxPooling2D, Conv2D, Flatten
import keras
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from tqdm import tqdm

label_data = np.load("label_data_new.npy")

label_name = np.load("label_name_new.npy")

data = np.load("data_new.npy")

labelncoder = LabelEncoder()
label_data = labelncoder.fit_transform(label_data)
label_data = to_categorical(label_data,52)
data = np.load("data_new.npy")
data = np.array(data)
data = data/255

x_train, x_test, y_train, y_test = train_test_split(data, label_data, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
Label_Actual = x_test.copy()


data_train = Label_Actual / 255
models = load_model("model_new_50.h5")
result = models.predict(x_test)
print(result)
predict_label = []

for i in result:
    max_predict = np.argmax(i)
    label_pre = label_name[max_predict]
    predict_label.append(label_pre)

predict_label = np.array(predict_label)
print(predict_label)

Actual_target = []
for j in y_test:
    max_actual = np.argmax(j)
    label_act = label_name[max_actual]
    Actual_target.append(label_act)

ac =y_test.copy()
pre = result.copy()

ac = ac.astype(str)
pre = pre.astype(str)
print(pre)

TP = []
FP = []
FN = []

cm = confusion_matrix(Actual_target, predict_label)
for _ in range(len(cm)):
    for __ in range(len(cm)):
        if  _== __:
            TP.append(cm[_][__])
        if  _!= __:
            if __ > _:
                FP.append(cm[_][__])
        if  _!= __:
            if __ < _:
                FN.append(cm[_][__])


df_cm = pd.DataFrame(cm, index = [i for i in label_name], columns=[i for i in label_name])
mask = cm < 1

TN = 5105 - (np.sum(TP) + np.sum(FP) + np.sum(FN))
print(TN)


data = [[np.sum(TP), np.sum(FP)], [np.sum(FN), TN]]
data = pd.DataFrame(data)
text_labels = np.array([[f"TP : {np.sum(TP)}", f"FP : {np.sum(FP)}"], [f"FN : {np.sum(FN)}", f"TN : {TN}"]])

heatmap = sns.heatmap(data, annot=text_labels, fmt='', cmap='Blues')

plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)

plt.figure(figsize=(52, 52))
sns.heatmap(df_cm, annot=True, cmap="rainbow_r", fmt="d", mask=mask)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()