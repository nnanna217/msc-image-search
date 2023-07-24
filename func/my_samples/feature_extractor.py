import os
import shutil
import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from numpy.linalg import norm
from keras import Sequential
from keras.layers import GlobalMaxPool2D
from keras.applications.resnet import ResNet50, preprocess_input
from tqdm import tqdm
from helper import select_images
import pickle


class FeatureExtractor:

    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
        self.model.trainable = False
        self.model = Sequential([
            self.model,
            GlobalMaxPool2D()
        ])

    def extract_features(self, img_path):
        img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
        image_arr = tf.keras.utils.img_to_array(img)
        # input_arr = np.array([image_arr])  # Convert single image to a batch.
        expanded_img_arr = np.expand_dims(image_arr, axis=0)
        preprocessed_img = preprocess_input(expanded_img_arr)
        predictions = self.model.predict(preprocessed_img).flatten()
        normalized_result = predictions / norm(predictions)

        return normalized_result
