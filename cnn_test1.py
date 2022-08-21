import tensorflow as tf
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report,confusion_matrix
from keras.preprocessing import image

train_dir = os.path.join("./archive/plant_dataset/train") #Train directory
test_dir = os.path.join("./archive/plant_dataset/valid") # Test Directory
tomato_files = os.path.join("./archive/plant_dataset/train/Tomato___Leaf_Mold")#directory for plotting sample images
tomato_image = os.listdir(tomato_files)  # Listing all the images from directory
pic_index = 20
next_plant = [os.path.join(tomato_files, fname)
                for fname in tomato_image[pic_index-2:pic_index]]
for i, img_path in enumerate(next_plant):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()


# Data Augmentation
# Global Variables
IMG_WIDTH = 224  # Width of image
IMG_HEIGHT = 224  # Height of Image
BATCH_SIZE = 32  # Batch Size
train_data_size = 70295
test_data_size = 17572
train_data = ImageDataGenerator(
                rescale=1./255,  # normalizing the input image
                rotation_range=0.2,
                vertical_flip=True,
                fill_mode='nearest')
test_data = ImageDataGenerator(
                rescale=1./255)
train_set = train_data.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode='categorical')
test_set = test_data.flow_from_directory(
                test_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                shuffle=False,
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
    model.add(dense(38, activation='softmax'))
    # Summary
    model.summary()

    return model

model = cnn_model(IMG_WIDTH, IMG_HEIGHT)

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['acc'])

history_model = model.fit_generator(train_set, steps_per_epoch=train_data_size // BATCH_SIZE+1, epochs=16,
        validation_data=test_set, validation_steps=test_data_size // BATCH_SIZE+1, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

#Save our model
model.save("first_one.h5")