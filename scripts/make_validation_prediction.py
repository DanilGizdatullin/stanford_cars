import matplotlib.pyplot as plt
import numpy as np

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from sklearn.metrics import confusion_matrix, accuracy_score

img_width, img_height = 227, 227
train_data_dir = "../data/data_for_nn/train"
validation_data_dir = "../data/data_for_nn/validation"
nb_train_samples = 5702
nb_validation_samples = 814
batch_size = 40
epochs = 50

sample_count = 814

model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
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

test_datagen = ImageDataGenerator(
    rescale=1./255
    # horizontal_flip=True,
    # fill_mode="nearest",
    # zoom_range=0.3,
    # width_shift_range=0.3,
    # height_shift_range=0.3,
    # rotation_range=30
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical",
    shuffle=False
)

print(validation_generator.filenames[0: 40])
# print(type(test_datagen))

valid_ans_pred = []
valid_ans = []

i = 0
for inputs_batch, labels_batch in validation_generator:
    ans = model_final.predict(inputs_batch)
    for j in range(inputs_batch.shape[0]):
        pred = np.argmax(ans[j])
        valid_ans_pred.append(pred)
        valid_ans.append(np.argmax(labels_batch[j]))
    i += 1
    if i * batch_size >= sample_count:
        break

# delete
    break
print(valid_ans_pred)
print(valid_ans)
# uncomment
# np.save('confusion_matrix.npy', confusion_matrix(valid_ans, valid_ans_pred))
# print(accuracy_score(valid_ans, valid_ans_pred))
