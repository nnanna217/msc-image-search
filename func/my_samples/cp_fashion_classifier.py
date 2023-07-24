import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Set the data directory
data = '/home/lunet/conce/Downloads/Codeproject/DeepFashion Custom/DeepFashion/Train'
test_data = '/home/lunet/conce/Downloads/Codeproject/DeepFashion Custom/DeepFashion/Test'
image_path = '/home/lunet/conce/Downloads/Codeproject/DeepFashion Custom/DeepFashion/Train/Blouse/img_00000003.jpg'
os.chdir(data)
batch_size = 3


def DataLoad(shape, preprocessing):
    """Create the training and validation datasets for a given image shape."""
    img_data = ImageDataGenerator(
        preprocessing_function=preprocessing,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.1,
        channel_shift_range=10.,
        horizontal_flip=True,
        validation_split=0.1,
    )

    height, width = shape

    # Load the training data
    train_dataset = img_data.flow_from_directory(
        os.getcwd(),
        target_size=(height, width),
        classes=['Blazer', 'Blouse', 'Cardigan', 'Dress', 'Jacket',
                 'Jeans', 'Jumpsuit', 'Romper', 'Shorts', 'Skirts', 'Sweater', 'Sweatpants',
                 'Tank', 'Tee', 'Top'],
        batch_size=batch_size,
        subset='training',
    )

    # Load the validation data
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


def plot_history(history, yrange):
    """Plot loss and accuracy as a function of the epoch, for the training and validation datasets."""
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)
    plt.legend(['train', 'validation'])

    plt.figure()

    # Plot training and validation loss per epoch
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.legend(['train', 'validation'])

    # Save the plots to a file
    plt.savefig('training_plots.png')


def plot_test_results(test_loss, test_acc):
    """Plot the test loss and test accuracy."""
    plt.figure()
    plt.bar(['Test Loss', 'Test Accuracy'], [test_loss, test_acc])
    plt.title('Test Loss and Test Accuracy')

    # Save the test plots to a file
    plt.savefig('test_plots.png')


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def get_class_string_from_index(index):
    for class_string, class_index in test_generator.class_indices.items():
        if class_index == index:
            return class_string

vgg16 = keras.applications.vgg16

# Load the VGG16 model without the top layers
conv_model = vgg16.VGG16(weights='imagenet', include_top=False)
conv_model.summary()

# Prepare the training and validation datasets
train_dataset, val_dataset = DataLoad((224, 224), preprocessing=vgg16.preprocess_input)
X_train, y_train = next(train_dataset)

image = np.expand_dims(plt.imread(image_path), 0)
plt.imshow(image[0])
gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15,
                         zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)
aug_iter = gen.flow(image)
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
plotImages(aug_images)

conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = keras.layers.Flatten()(conv_model.output)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
predictions = keras.layers.Dense(15, activation='softmax')(x)
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
full_model.summary()

for layer in conv_model.layers:
    # Freeze the layers of the pre-trained model
    layer.trainable = False

full_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adamax(lr=0.001), metrics=['acc'])

history = full_model.fit_generator(
    train_dataset,
    validation_data=val_dataset,
    workers=0,
    epochs=10,
)

# Plot the training and validation history
plot_history(history, yrange=(0.9, 1))

# Prepare the test data
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_data, target_size=(224, 224), batch_size=3,
                                                  class_mode='categorical')
# Evaluate the model on the test data
test_results = full_model.evaluate(test_generator)

# Print the test loss and test accuracy
print("Test loss, Test accuracy:", test_results)

# Plot the test loss and test accuracy
test_loss = test_results[0]
test_acc = test_results[1]

# Plot the test results
plot_test_results(test_loss, test_acc)

# Load the saved training plots
training_plots = plt.imread('training_plots.png')
# Display the training plots
plt.figure()
plt.imshow(training_plots)
plt.title('Training Plots')
plt.axis('off')
plt.show()

# Load the saved test plots
test_plots = plt.imread('test_plots.png')

# Display the test plots
plt.figure()
plt.imshow(test_plots)
plt.title('Test Plots')
plt.axis('off')
plt.show()

# Section 3 -

test_generator = test_datagen.flow_from_directory(test_data, target_size=(224, 224), batch_size=7, class_mode='categorical')
X_test, y_test = next(test_generator)
X_test = X_test / 255
image = X_test[2]
true_index = np.argmax(y_test[2])

plt.imshow(image)
plt.axis('off')

# Save the plot in a file
plt.savefig('test_image.png')

# Load the saved plot
saved_image = plt.imread('test_image.png')

# Display the loaded plot
plt.imshow(saved_image)
plt.axis('off')
plt.show()

# Expand the validation image to (1, 224, 224, 3) before predicting the label
prediction_scores = full_model.predict(np.expand_dims(image, axis=0))
predicted_index = np.argmax(prediction_scores)

print("True label: " + get_class_string_from_index(true_index))
print("Predicted label: " + get_class_string_from_index(predicted_index))
