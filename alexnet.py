import pandas as pd
import tensorflow as tf
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
import shutil
# dataset = pd.read_csv("./NIPA_하반기 경진대회_사전검증/train/train.tsv", delimiter='\t', header=None)
#
# class_name = []
# for i in range(len(dataset)):
#     class_name.append(str(dataset[1][i]) + '_' + str(dataset[2][i]))
#
# dataset['class'] = class_name
# print(dataset)

dataset = pd.read_csv('./NIPA_하반기 경진대회_사전검증/sample_file.tsv', header=None, sep="\t")
print(dataset)

def make_dir():
    for i in range(len(dataset)):
        try:
            if not (os.path.isdir('./NIPA_하반기 경진대회_사전검증/process_train/%s' % dataset[3][i])):
                os.makedirs('./NIPA_하반기 경진대회_사전검증/process_train/%s' % dataset[3][i])
        except OSError as e:
                print("Failed to create directory!!!!!")

        file = './NIPA_하반기 경진대회_사전검증/train/%s' % dataset[0][i]
        copy_file = './NIPA_하반기 경진대회_사전검증/process_train/%s/%s' % (dataset[3][i], dataset[0][i])

        shutil.copy2(file, copy_file)


train_dir = os.path.join("./NIPA_하반기 경진대회_사전검증/process_train")  # Train directory


# Data Augmentation
# Global Variables
IMG_WIDTH = 224  # Width of image
IMG_HEIGHT = 224  # Height of Image
BATCH_SIZE = 32  # Batch Size
train_data_size = 12800
valid_set_size = 3200
train_data = ImageDataGenerator(
                rescale=1./255,  # normalizing the input image
                shear_range=0.2,
                zoom_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
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
    classifier = Sequential()

    # Convolution Step 1
    classifier.add(Convolution2D(96, 11, strides=(4, 4), padding='valid', input_shape=(224, 224, 3), activation='relu'))

    # Max Pooling Step 1
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 2
    classifier.add(Convolution2D(256, 11, strides=(1, 1), padding='valid', activation='relu'))

    # Max Pooling Step 2
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Convolution Step 3
    classifier.add(Convolution2D(384, 3, strides=(1, 1), padding='valid', activation='relu'))
    classifier.add(BatchNormalization())

    # Convolution Step 4
    classifier.add(Convolution2D(384, 3, strides=(1, 1), padding='valid', activation='relu'))
    classifier.add(BatchNormalization())

    # Convolution Step 5
    classifier.add(Convolution2D(256, 3, strides=(1, 1), padding='valid', activation='relu'))

    # Max Pooling Step 3
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    classifier.add(BatchNormalization())

    # Flattening Step
    classifier.add(Flatten())

    # Full Connection Step
    classifier.add(Dense(units=4096, activation='relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=4096, activation='relu'))
    classifier.add(Dropout(0.4))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=1000, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=20, activation='softmax'))
    classifier.summary()

    return classifier

model = cnn_model(IMG_WIDTH, IMG_HEIGHT)

# Compiling the Model
from keras import optimizers
# Compiling model
model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=50,
        validation_data=valid_set, validation_steps=valid_set_size // BATCH_SIZE+1, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

#Save our model
model.save("first_one.h5")