# Code adapted from https://www.codeproject.com/Articles/5297227/Deep-Learning-for-Fashion-Classification
# import tensorflow.keras as keras

import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

data = '/home/lunet/conce/Downloads/Codeproject/DeepFashion Custom/DeepFashion/Train'
os.chdir(data)
batch_size = 3


def DataLoad(shape, preprocessing):
    """Create the training and validation datasets for a given image shape."""
    img_data = ImageDataGenerator(
        preprocessing_function=preprocessing,
        horizontal_flip=True,
        validation_split=0.1,
    )

    height, width = shape

    train_dataset = img_data.flow_from_directory(
        os.getcwd(),
        target_size=(height, width),
        classes=['Blazer', 'Blouse', 'Cardigan', 'Dress', 'Jacket',
                 'Jeans', 'Jumpsuit', 'Romper', 'Shorts', 'Skirts', 'Sweater', 'Sweatpants'
            , 'Tank', 'Tee', 'Top'],
        batch_size=batch_size,
        subset='training',
    )

    val_dataset = img_data.flow_from_directory(
        os.getcwd(),
        target_size=(height, width),
        classes=['Blazer', 'Blouse', 'Cardigan', 'Dress', 'Jacket',
                 'Jeans', 'Jumpsuit', 'Romper', 'Shorts', 'Skirts', 'Sweater',
                 'Sweatpants', 'Tank', 'Tee', 'Top'],
        batch_size=batch_size,
        subset='validation'
    )

    return train_dataset, val_dataset


vgg16 = keras.applications.vgg16
conv_model = vgg16.VGG16(weights='imagenet', include_top=False)
conv_model.summary()

train_dataset, val_dataset = DataLoad((224, 224), preprocessing=vgg16.preprocess_input)
# Function for plots images with labels within jupyter notebook
X_train, y_train = next(train_dataset)

# Load ImageNet weights of this network, to be used during the transfer learning
conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# flatten the output of the convolutional part:
x = keras.layers.Flatten()(conv_model.output)

# three hidden layers
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)

# final softmax layer with 15 categories
predictions = keras.layers.Dense(15, activation='softmax')(x)

# creating the full model:
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
full_model.summary()

for layer in conv_model.layers:
    layer.trainable = False

full_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adamax(lr=0.001), metrics=['acc'])

history = full_model.fit_generator(
    train_dataset,
    validation_data=val_dataset,
    workers=0,
    epochs=7,
)


def plot_history(history, yrange):
    """Plot loss and accuracy as a function of the epoch, for the training and validation datasets.
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)

    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

    plt.show()


plot_history(history, yrange=(0.9, 1))

test_data = '/home/lunet/conce/Downloads/Codeproject/DeepFashion Custom/DeepFashion/Train'
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(test_data, target_size=(224, 224), batch_size=3, class_mode='categorical')
# X_test, y_test = next(test_generator)

test_results = full_model.evaluate(test_generator)
print("test loss, test acc:", test_results)