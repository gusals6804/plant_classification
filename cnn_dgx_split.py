import pandas as pd
import tensorflow as tf
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

train_dir = os.path.join("./process_split_train2/train")  # Train directory
valid_dir = os.path.join("./process_split_train2/val")

# Data Augmentation
# Global Variables
IMG_WIDTH = 244  # Width of image
IMG_HEIGHT = 244  # Height of Image
BATCH_SIZE = 64  # Batch Size
train_data_size = 14400
valid_data_size = 1600
train_data = ImageDataGenerator(
                rescale=1. / 255,  # normalizing the input image
                horizontal_flip=True,
                vertical_flip=True,
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
    # Defining all layers.
    dense = tf.keras.layers.Dense
    conv = tf.keras.layers.Conv2D
    batch = tf.keras.layers.BatchNormalization
    max_pooling = tf.keras.layers.MaxPooling2D
    flatten = tf.keras.layers.Flatten()
    dropout = tf.keras.layers.Dropout(0.3)
    dropout2 = tf.keras.layers.Dropout(0.5)
    # Sequential Model
    model = tf.keras.Sequential()
    # 1st layer
    model.add(conv(32, (1, 1), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    # 1st layer
    model.add(conv(64, (3, 3), padding='same', activation='relu'))
    model.add(conv(64, (3, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    # 2nd layer
    model.add(conv(128, (3, 3), padding='same', activation='relu'))
    model.add(conv(128, (3, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    # 3nd layer
    model.add(conv(256, (3, 3), padding='same', activation='relu'))
    model.add(conv(256, (3, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    # 3nd layer
    model.add(conv(512, (3, 3), padding='same', activation='relu'))
    model.add(conv(512, (3, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    model.add(conv(1024, (3, 3), padding='same', activation='relu'))
    model.add(conv(1024, (3, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))

    model.add(flatten)

    model.add(dense(1024, activation='relu'))
    model.add(dropout2)
    model.add(dense(512, activation='relu'))
    model.add(dropout2)
    model.add(dense(128, activation='relu'))
    model.add(dropout2)
    model.add(dense(32, activation='relu'))
    # Output Layer
    model.add(dense(20, activation='softmax'))
    # Summary
    model.summary()

    return model

model = cnn_model(IMG_WIDTH, IMG_HEIGHT)

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['acc'])


modelpath = "./models/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=modelpath,
                                                  monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=200,
                                    validation_data=valid_set, validation_steps=valid_data_size // BATCH_SIZE+1, verbose=1,
                                    callbacks=[early_stopping_callback, checkpointer])

#Save our model
model.save("16_32_64_128_cnn_50_split_3layer_adam_ver_ho_256_64.h5")