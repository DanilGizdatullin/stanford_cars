import numpy as np
import os

from scipy import misc

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

dict_of_classes = {i: val for i, val in enumerate(sorted(os.listdir('../data/data_for_nn/train')))}

img_width, img_height = 227, 227
validation_data_dir = "../data/data_for_nn/validation"
nb_train_samples = 5702
nb_validation_samples = 814
batch_size = 40
epochs = 50

model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 17 layers.
number_of_layers = 0
for layer in model.layers[:17]:
    print(layer)
    layer.trainable = False

model.summary()

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(196, activation="softmax")(x)

model_final = Model(input=model.input, output=predictions)

model_final.load_weights('vgg16_1.h5')

model_final.compile(loss="categorical_crossentropy",
                    optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
                    metrics=["accuracy"])

name_of_directory = '../new_photos/'
for file_name in os.listdir(name_of_directory):
    img = misc.imread(os.path.join(name_of_directory, file_name))
    img = misc.imresize(img, (img_width, img_height, 3))
    img = img / 255.0
    img = img.reshape((1,) + img.shape)

    valid_ans_pred = []
    valid_ans = []

    ans = model_final.predict(img)
    print(file_name, dict_of_classes[np.argmax(ans)], np.max(ans))
