import pandas as pd
import tensorflow as tf
import keras
import os
import numpy as np

from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add

from functools import partial
from keras_preprocessing.image import ImageDataGenerator

train_dir = os.path.join("./process_split_train/train")  # Train directory
valid_dir = os.path.join("./process_split_train/val")

# Data Augmentation
# Global Variables
IMG_WIDTH = 256  # Width of image
IMG_HEIGHT = 256  # Height of Image
BATCH_SIZE = 16  # Batch Size
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
    input_tensor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), dtype='float32', name='input')

    def conv1_layer(x):
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (7, 7), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

        return x

    def conv2_layer(x):
        x = MaxPooling2D((3, 3), 2)(x)

        shortcut = x

        for i in range(3):
            if (i == 0):
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
                x = BatchNormalization()(x)
                shortcut = BatchNormalization()(shortcut)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

        return x

    def conv3_layer(x):
        shortcut = x

        for i in range(4):
            if (i == 0):
                x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = BatchNormalization()(x)
                shortcut = BatchNormalization()(shortcut)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

        return x

    def conv4_layer(x):
        shortcut = x

        for i in range(6):
            if (i == 0):
                x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = BatchNormalization()(x)
                shortcut = BatchNormalization()(shortcut)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

        return x

    def conv5_layer(x):
        shortcut = x

        for i in range(3):
            if (i == 0):
                x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
                shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
                x = BatchNormalization()(x)
                shortcut = BatchNormalization()(shortcut)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)

                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

        return x

    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)

    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(20, activation='softmax')(x)

    resnet50 = Model(input_tensor, output_tensor)
    resnet50.summary()

    return resnet50

model = cnn_model(IMG_WIDTH, IMG_HEIGHT)

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['acc'])


modelpath = "./models/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=modelpath,
                                                  monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)


history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=50,
                                    validation_data=valid_set, validation_steps=valid_data_size // BATCH_SIZE+1, verbose=1,
                                    callbacks=[early_stopping_callback, checkpointer])

#Save our model

model.save("resnet.h5")