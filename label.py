import pandas as pd
import tensorflow as tf
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix
from keras.preprocessing import image
import shutil

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

labels_values, no_of_images = np.unique(train_set.classes, return_counts=True)
dict(zip(train_set.class_indices, no_of_images))
labels = train_set.class_indices
labels = { v:k for k,v in labels.items()}  # Flipping keys and values
values_lbl = list(labels.values())  # Taking out only values from dictionary
print(labels)