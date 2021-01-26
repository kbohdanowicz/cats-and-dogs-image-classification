import os
import random

from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model, load_model
from data_processing import DATASET_DIRECTORY
from utils import measure_time
from keras.preprocessing.image import ImageDataGenerator
from seaborn import heatmap
from sklearn import metrics
from pathlib import Path

import matplotlib.pyplot as plt

IMAGE_SIZE = 200


# noinspection DuplicatedCode
def get_cnn_model(type: int):
    model = None

    # https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
    #
    # One layer
    if type == 1:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))

        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # Three layers
    elif type == 2:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))

        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # Three layers with dropout
    elif type == 3:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # VGG16
    elif type == 4:
        model = VGG16(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

        for layer in model.layers:
            layer.trainable = False

        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)

        model = Model(inputs=model.inputs, outputs=output)

        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


def get_cnn_callbacks() -> list:
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                                mode="min",
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    early_stopping = EarlyStopping(monitor="loss",
                                   mode="min",
                                   patience=5,
                                   restore_best_weights=True,
                                   verbose=1)

    return [early_stopping, learning_rate_reduction]


def get_cnn_train_test_data() -> tuple:
    train_data_gen = ImageDataGenerator(rescale=1.0 / 255.0,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        horizontal_flip=True)

    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

    _batch_size = 16

    train_set = train_data_gen.flow_from_directory(f'{DATASET_DIRECTORY}train/',
                                                   target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                   batch_size=_batch_size,
                                                   class_mode='binary')

    test_set = test_data_gen.flow_from_directory(f'{DATASET_DIRECTORY}test/',
                                                 target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                 batch_size=_batch_size,
                                                 class_mode='binary')

    return train_set, test_set


def save_plot(history, type: int):
    title = ''
    if type == 1:
        title = 'One layer'
    if type == 2:
        title = 'Three layers'
    if type == 3:
        title = 'Three layers with dropout'
    if type == 4:
        title = 'VGG16'

    pyplot.title(title)
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.legend(loc='upper center', bbox_to_anchor=(0.85, -0.05), ncol=2)
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Accuracy')
    pyplot.savefig(f'plot/model_{type}_plot.png')
    pyplot.close()


def save_model(model, name: str):
    model.save(f'model/{name}.h5')


@measure_time
def run_cnn(type: int):
    model = get_cnn_model(type)

    (train_set, test_set) = get_cnn_train_test_data()

    _steps_per_epoch = 200
    _validation_steps = 200
    _epochs = 30

    history = model.fit_generator(train_set,
                                  steps_per_epoch=_steps_per_epoch,
                                  validation_data=test_set,
                                  validation_steps=_validation_steps,
                                  epochs=_epochs,
                                  callbacks=get_cnn_callbacks(),
                                  max_queue_size=32,
                                  workers=16)

    (loss, accuracy) = model.evaluate(test_set, steps=len(test_set), verbose=2)
    model_name = f'model_{type}_acc_{accuracy}'
    save_model(model, model_name)
    save_plot(history, type)


@measure_time
def cnn_predict(model_name: str, type: int):
    model = load_model(f'model/{model_name}')

    cat_image_paths = list(Path(f'{DATASET_DIRECTORY}test/cats/').rglob("*.jpg"))
    dog_image_paths = list(Path(f'{DATASET_DIRECTORY}test/dogs/').rglob("*.jpg"))

    random.shuffle(cat_image_paths)
    random.shuffle(dog_image_paths)

    test_image_paths = []

    half_of_test_images = 50

    for (i, path) in enumerate(cat_image_paths):
        if i >= half_of_test_images:
            break
        test_image_paths.append(path)

    for (i, path) in enumerate(dog_image_paths):
        if i >= half_of_test_images:
            break
        test_image_paths.append(path)

    random.shuffle(test_image_paths)

    image_labels = []
    predictions = []

    for (i, path) in enumerate(test_image_paths):
        if i >= half_of_test_images * 2:
            break

        path = str(path)

        img = image.load_img(path, target_size=(200, 200))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)

        result = model.predict(img_batch)

        label = path.split(os.path.sep)[-1].split(".")[0]
        image_labels.append(label)

        if result[0][0] >= 0.5:
            predictions.append('dog')
        else:
            predictions.append('cat')

    tick_labels = ['cat', 'dog']
    ax = heatmap(metrics.confusion_matrix(image_labels, predictions),
                 xticklabels=tick_labels,
                 yticklabels=tick_labels,
                 annot=True,
                 fmt='d')

    title = ''
    if type == 1:
        title = 'One layer'
    if type == 2:
        title = 'Three layers'
    if type == 3:
        title = 'Three layers with dropout'
    if type == 4:
        title = 'VGG16'

    plt.title(title)
    plt.show()
