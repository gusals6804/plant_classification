import pandas as pd
import tensorflow as tf
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


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
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), kernel_size=(3, 3), filters=32, padding='same',
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=20, activation='softmax'))

    # Summary
    model.summary()

    return model

model = cnn_model(IMG_WIDTH, IMG_HEIGHT)

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['acc'])


modelpath = "./models/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=modelpath,
                                                  monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=50,
                                    validation_data=valid_set, validation_steps=valid_data_size // BATCH_SIZE+1, verbose=1,
                                    callbacks=[early_stopping_callback, checkpointer])

#Save our model
model.save("vgg_50_split.h5")