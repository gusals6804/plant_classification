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
BATCH_SIZE = 64  # Batch Size
train_data_size = 12800
valid_set_size = 3200
train_data = ImageDataGenerator(
                rescale=1./255,  # normalizing the input image
                rotation_range=0.2,
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
    # 2nd Layer
    model.add(conv(32, (3, 3), padding='same', activation='relu'))
    model.add(max_pooling(2, 2))
    # Flatten Layer
    model.add(flatten)
    # 1st Hidden Layer
    model.add(dense(512, activation='relu',))
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

history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=16,
        validation_data=valid_set, validation_steps=valid_set_size // BATCH_SIZE+1, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

#Save our model
model.save("first_one.h5")