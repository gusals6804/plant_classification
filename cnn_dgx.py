import pandas as pd
import tensorflow as tf
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report,confusion_matrix
from keras.preprocessing import image
import shutil


train_dir = os.path.join("./process_train")  # Train directory


# Data Augmentation
# Global Variables
IMG_WIDTH = 224  # Width of image
IMG_HEIGHT = 224  # Height of Image
BATCH_SIZE = 32  # Batch Size
train_data_size = 12800
valid_set_size = 3200
train_data = ImageDataGenerator(
                rescale=1./255,  # normalizing the input image
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=[0.8, 2.0],
                horizontal_flip=True,
                vertical_flip=True,
                validation_split=0.2,
                fill_mode='nearest')
train_set = train_data.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                shuffle=True,
                subset="training",
                batch_size=BATCH_SIZE,
                class_mode='categorical')
valid_set = train_data.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                shuffle=True,
                subset="validation",
                batch_size=BATCH_SIZE,
                class_mode='categorical')



def cnn_model(IMG_WIDTH, IMG_HEIGHT):
    # Defining all layers.
    dense = tf.keras.layers.Dense
    conv = tf.keras.layers.Conv2D
    max_pooling = tf.keras.layers.MaxPooling2D
    flatten = tf.keras.layers.Flatten()
    dropout = tf.keras.layers.Dropout(0.2)
    # Sequential Model
    model = tf.keras.Sequential()
    # 1st layer
    model.add(conv(16, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    # 2nd layer
    model.add(conv(32, (3, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    # 3nd layer
    model.add(conv(64, (3, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    # Flatten Layer
    model.add(flatten)
    # 1st Hidden Layer
    model.add(dense(512, activation='relu', ))
    model.add(dropout)
    # 2nd Hidden Layer
    model.add(dense(256, activation='relu'))
    # Output Layer
    model.add(dense(20, activation='softmax'))
    # Summary
    model.summary()

    return model

model = cnn_model(IMG_WIDTH, IMG_HEIGHT)

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['acc'])

history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=100,
        validation_data=valid_set, validation_steps=valid_set_size // BATCH_SIZE+1, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)])

#Save our model
model.save("first_one_100.h5")