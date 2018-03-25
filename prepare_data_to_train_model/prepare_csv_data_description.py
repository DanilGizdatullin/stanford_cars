# encoding:utf8

from scipy.io import loadmat
import numpy as np
import os

PATH_TO_IMAGES = "/Users/dgizdatullin/Downloads/cars_train/"
PATH_TO_DEVKIT = "/Users/dgizdatullin/Downloads/devkit/"

mat_train = loadmat(os.path.join(PATH_TO_DEVKIT, 'cars_train_annos.mat'))
mat_test = loadmat(os.path.join(PATH_TO_DEVKIT, 'cars_test_annos.mat'))
meta = loadmat(os.path.join(PATH_TO_DEVKIT, 'cars_meta.mat'))

labels = list()
for l in meta['class_names'][0]:
    labels.append(l[0])

train = list()
for example in mat_train['annotations'][0]:
    label = labels[example[-2][0][0] - 1]
    image = example[-1][0]
    train.append((image, label))

test = list()
for example in mat_test['annotations'][0]:
    image = example[-1][0]
    test.append(image)

validation_size = int(len(train) * 0.10)
test_size = int(len(train) * 0.20)

validation = train[:validation_size].copy()
np.random.shuffle(validation)
train = train[validation_size:]

test = train[:test_size].copy()
np.random.shuffle(test)
train = train[test_size:]

if not os.path.isdir("../data/"):
    os.mkdir('../data')
bucket_path = PATH_TO_IMAGES

with open('../data/cars_data.csv', 'w+') as f:
    [f.write('TRAIN,%s%s,%s\n' % (bucket_path, img, lab)) for img, lab in train]
    [f.write('VALIDATION,%s%s,%s\n' % (bucket_path, img, lab)) for img, lab in validation]
    [f.write('TEST,%s%s\n' % (bucket_path, img)) for img, _ in test]
