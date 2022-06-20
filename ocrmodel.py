import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from tensorflow import keras
num=tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test)=num.load_data()
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

for train in range(len(x_train)):
  for row in range(28):
    for x in range(28):
      if x_train[train][row][x] !=0:
          x_train[train][row][x] = 1
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
from tensorflow.python.eager.monitoring import Metric
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'] )
model.fit(x_train, y_train, epochs=200)
model.save('D:\OCR model\epic.model')
print('model_save')