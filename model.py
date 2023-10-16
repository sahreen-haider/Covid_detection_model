# importing the necessary libraries

import tensorflow as tf                     # importing tensorflow for buliding the model and using Deep Learning Frameworks
from tensorflow.keras.models import Model               
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd                 
import os
import cv2
import matplotlib.pyplot as plt


os.environ['SM_FRAMEWORK'] = 'tf.keras'

import segmentation_models as sm

# resizing all the images
IMAGE_SIZE = (100,100)
path = '/Users/sahreenhaider/Downloads/New_Data_CoV2'
data = []


def data_loader():                      # fn for loading the data
  main_folder = 0
  folder = 0
  sub_folder = 0
  c = 0
  try:
    for folder in os.listdir(path):
      try:
        sub_path=path+"/"+folder
        for folder2 in os.listdir(sub_path):
          try:
            sub_path2=sub_path+"/"+folder2
            for img in os.listdir(sub_path2):
              image_path=sub_path2+"/"+img
              img_arr=cv2.imread(image_path)
              img_arr=cv2.resize(img_arr,IMAGE_SIZE)
              data.append(img_arr)
          except Exception as E:
            c+=1
      except Exception as E:
        sub_folder += 1
  except Exception as E:
    main_folder += 1   

data_loader()

X = np.array(data)

# normalizing the input
x = X/255

# using datagen to agument the existing input data to make the prediction labels
datagen = ImageDataGenerator(rescale = 1/255)
dataset = datagen.flow_from_directory(path,
                                    target_size = IMAGE_SIZE,
                                    batch_size = 32,
                                    class_mode = 'sparse')

# classes of predict labels
dataset.class_indices

y = dataset.classes

# splitting the data into train, validation, test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


from tensorflow._api.v2.nn import dropout
model = tf.keras.Sequential()

# convoltuion layer
model.add(tf.keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, input_shape = x_train[0].shape))
# pooling layer
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.BatchNormalization())


# convolution layer
model.add(tf.keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu))
# pooling layer
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.BatchNormalization())

# convolution layer
model.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu))
# pooling layer
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.BatchNormalization())

# convolution layer
model.add(tf.keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu))
# pooling layer
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.BatchNormalization())

# input layer
model.add(tf.keras.layers.Flatten())


# output layer
model.add(tf.keras.layers.Dense(3, activation = tf.nn.softmax))

model.summary()


# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, start_from_epoch = 7, patience=5)

record = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, callbacks=[early_stop], shuffle=True, verbose=0)


model.save('/Users/sahreenhaider/Documents/Covid_detection_model/model.h5')



