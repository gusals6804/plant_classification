import tensorflow as tf
import os
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from keras.preprocessing import image
import natsort

train_dir = os.path.join("./NIPA_하반기 경진대회_사전검증/process_split_train/train")  # Train directory
valid_dir = os.path.join("./NIPA_하반기 경진대회_사전검증/process_split_train/val")

model_name = '37-0.1073.hdf5'

# Data Augmentation
# Global Variables
IMG_WIDTH = 244  # Width of image
IMG_HEIGHT = 244  # Height of Image
BATCH_SIZE = 32  # Batch Size
train_data_size = 12800
valid_data_size = 3200
train_data = ImageDataGenerator(
                rescale=1./255,  # normalizing the input image
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=[0.8, 2.0],
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest')
val_data = ImageDataGenerator(
                rescale=1./255)
train_set = train_data.flow_from_directory(
                train_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                shuffle=False,
                batch_size=BATCH_SIZE,
                class_mode='categorical')
valid_set = val_data.flow_from_directory(
                valid_dir,
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                shuffle=False,
                batch_size=BATCH_SIZE,
                class_mode='categorical')

labels_values, no_of_images = np.unique(train_set.classes, return_counts=True)
dict(zip(train_set.class_indices, no_of_images))
labels = train_set.class_indices
labels = {v: k for k, v in labels.items()}  # Flipping keys and values
print(labels)
values_lbl = list(labels.values())  # Taking out only values from dictionary

model = tf.keras.models.load_model('./%s' % model_name)

y_test = valid_set.classes
#predicting our model with test dataset i.e. unseen dataset
pred = model.predict_generator(valid_set, len(valid_set), verbose=1,).argmax(axis=1)
#Classification report of every label
print(classification_report(y_test, pred))


path_dir = './NIPA_하반기 경진대회_사전검증/test'
file_list = os.listdir(path_dir)
file_list = natsort.natsorted(file_list)

acc = []
image_name = []
plant = []
dis = []
bad_image = []

for i in file_list:
    bad = []
    img = image.load_img('./NIPA_하반기 경진대회_사전검증/test/%s' % i, target_size=(IMG_WIDTH, IMG_HEIGHT, 3))
    img = image.img_to_array(img)
    img = img / 255
    proba = model.predict(img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3))
    top_1 = proba.argmax()
    print(i)
    print("Label: {}".format(labels[top_1]))
    print("Accuracy: {:2}".format(proba[0][top_1] * 100))
    plt.imshow(img)
    plt.title("Label: %s, Accuracy: %02d" % (labels[top_1], proba[0][top_1] * 100))
    plt.axis('Off')
    acc.append(proba[0][top_1] * 100)
    #plt.show()

    image_name.append(i)
    plant.append(labels[top_1].split('_')[0])
    dis.append(labels[top_1].split('_')[1])

    if proba[0][top_1] * 100 < 50:
        bad_image.append([i, labels[top_1]])

df = pd.DataFrame({"image": image_name, "plant": plant, "dis": dis})
df.to_csv('%s.tsv' % model_name, index=False, header=None, sep="\t")

acc = np.array(acc)
print(acc.mean())
print(bad_image)