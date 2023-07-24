import os
import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adamax
from keras.preprocessing.image import ImageDataGenerator
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband


class MyHyperModel(HyperModel):
    def __init__(self, num_classes):
        """
        Initialize the MyHyperModel class.

        Args:
            num_classes (int): Number of classes in the classification task.
        """
        self.num_classes = num_classes

    def build(self, hp):
        """
        Build the hypermodel by defining the model architecture.

        Args:
            hp (HyperParameters): HyperParameters object.

        Returns:
            tensorflow.keras.Model: Constructed model.

        """
        # Load the VGG16 model without the top layers
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

        # Freeze the layers of the pre-trained model
        base_model.trainable = False

        x = base_model.output
        x = Flatten()(x)

        # Add a dense layer with a variable number of units
        x = Dense(
            hp.Int('dense_units', min_value=64, max_value=512, step=64, default=256),
            activation='relu'
        )(x)

        # Add the final dense layer for classification
        x = Dense(self.num_classes, activation='softmax')(x)

        # Create the model with the defined architecture
        model = Model(inputs=base_model.input, outputs=x)

        # Compile the model with a variable learning rate
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adamax(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            metrics=['accuracy']
        )

        return model


data_dir = '/home/lunet/conce/Downloads/Codeproject/DeepFashion Custom/DeepFashion/Train'
test_dir = '/home/lunet/conce/Downloads/Codeproject/DeepFashion Custom/DeepFashion/Test'
batch_size = 32
num_classes = 15
input_shape = (224, 224, 3)


def load_data(data_dir, input_shape):
    """
    Load and preprocess the data using ImageDataGenerator.

    Args:
        data_dir (str): Directory path containing the data.
        input_shape (tuple): Input shape of the images (height, width, channels).

    Returns:
        tuple: Tuple containing train_generator and val_generator.

    """
    # Create an ImageDataGenerator for data preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Rescale pixel values to [0, 1]
        validation_split=0.1,  # Split the data into train and validation sets
        preprocessing_function=keras.applications.vgg16.preprocess_input  # Apply VGG16 preprocessing
    )

    # Generate the training data from the directory
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Subset of the data (training set)
    )

    # Generate the validation data from the directory
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Subset of the data (validation set)
    )

    return train_generator, val_generator


def objective(hp):
    """
    Define the objective function for hyperparameter tuning.

    Args:
        hp (HyperParameters): HyperParameters object.

    Returns:
        tuple: Tuple containing the best hyperparameters and the best model.

    """
    # Create an instance of the custom hypermodel
    hypermodel = MyHyperModel(num_classes=num_classes)

    # Load and preprocess the data
    train_generator, val_generator = load_data(data_dir, input_shape)

    # Create a Hyperband tuner with the hypermodel
    tuner_model = Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=10,
        directory='hyperband',
        project_name='fashion'
    )

    # Search for the best hyperparameters using the training and validation data
    tuner_model.search(train_generator, validation_data=val_generator, verbose=2)

    # Get the best hyperparameters and the best model from the tuner
    best_hps = tuner_model.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner_model.get_best_models(1)[0]

    # Print the summary of the best model
    best_model.summary()

    return best_hps, best_model


best_hps, best_model = objective()

print("Best hyperparameters:")
print(best_hps.values)

print("Best model summary:")
best_model.summary()
