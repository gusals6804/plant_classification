import pandas as pd
import tensorflow as tf
import os
import shutil
import splitfolders

def class_tsv():
    dataset = pd.read_csv("./NIPA_하반기 경진대회_사전검증/train/train.tsv", delimiter='\t', header=None)

    class_name = []
    for i in range(len(dataset)):
        class_name.append(str(dataset[1][i]) + '_' + str(dataset[2][i]))

    dataset['class'] = class_name
    print(dataset)
    dataset.to_csv('train_class.tsv', index=False, header=None, sep="\t")

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

def split_folder():
    splitfolders.ratio('./NIPA_하반기 경진대회_사전검증/process_train/',
                       output="./NIPA_하반기 경진대회_사전검증/process_split_train2", seed=None, ratio=(.9, .1))


class_tsv()
dataset = pd.read_csv('./train_class.tsv', header=None, sep="\t")
print(dataset)
split_folder()