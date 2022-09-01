import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping


train_dir = os.path.join("./process_split_train/train")  # Train directory
valid_dir = os.path.join("./process_split_train/val")

# Data Augmentation
# Global Variables
IMG_WIDTH = 256  # Width of image
IMG_HEIGHT = 256  # Height of Image
BATCH_SIZE = 32  # Batch Size
train_data_size = 12800
valid_data_size = 3200
train_data = ImageDataGenerator(
                rescale=1. / 255,  # normalizing the input image
                vertical_flip=True,
                horizontal_flip=True,
                fill_mode='nearest')
valid_data = ImageDataGenerator(
                rescale=1./255)
train_set = train_data.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode='categorical')
valid_set = valid_data.flow_from_directory(
                valid_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                shuffle=False,
                class_mode='categorical')



def cnn_model(IMG_WIDTH, IMG_HEIGHT):
    classifier = Sequential()

    classifier.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                          input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'))
    classifier.add(MaxPool2D((2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    classifier.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
    classifier.add(MaxPool2D((2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    classifier.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    classifier.add(BatchNormalization())

    classifier.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    classifier.add(BatchNormalization())

    classifier.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

    classifier.add(MaxPool2D((2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    classifier.add(Flatten())

    classifier.add(Dense(4096, activation='relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(4096, activation='relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(1000, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(20, activation='softmax'))

    classifier.summary()


    return classifier

model = cnn_model(IMG_WIDTH, IMG_HEIGHT)

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['acc'])


modelpath = "./models/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=modelpath,
                                                  monitor='val_loss', verbose=1, save_best_only=True)

# early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
#
#
# history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=200,
#                                     validation_data=valid_set, validation_steps=valid_data_size // BATCH_SIZE+1, verbose=1,
#                                     callbacks=[early_stopping_callback, checkpointer])

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=200,
                                    validation_data=valid_set, validation_steps=valid_data_size // BATCH_SIZE+1, verbose=1,
                                    callbacks=[early_stopping_callback, checkpointer])


#Save our model
model.save("alexnet3.h5")