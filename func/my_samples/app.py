import os
from helper import select_images
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
import pickle

# source_folder = '/home/lunet/conce/Documents/deepfashion2/data/train'
train_src_folder = '/home/lunet/conce/Documents/deepfashion2/data/train/image'
destination_folder = '/home/lunet/conce/PycharmProjects/fashion-reccomender-system/data/images'

# Copy subset of training image into the project folder
# select_images(train_src_folder, destination_folder + '/train', limit=100000)

print("===================== Done===================")
print("===================== CNN Phase===================")
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([
    model,
    GlobalMaxPool2D()
])


print(model.summary())


def extract_features(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    image_arr = tf.keras.utils.img_to_array(img)
    # input_arr = np.array([image_arr])  # Convert single image to a batch.
    expanded_img_arr = np.expand_dims(image_arr, axis=0)
    preprocessed_img = preprocess_input(expanded_img_arr)
    predictions = model.predict(preprocessed_img).flatten()
    normalized_result = predictions / norm(predictions)

    return normalized_result


filepath = destination_folder + "/train"
filenames = []

for file in os.listdir(filepath):
    filenames.append(os.path.join(filepath, file))
# print(len(filenames))
# print(filenames[0:5])

print("=====================Feature extraction ===================")
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))
print(np.array(feature_list).shape)

pickle.dump(feature_list, open('static/index/embeddings_100000.pkl', 'wb'))
pickle.dump(filenames, open('static/index/filenames_100000.pkl', 'wb'))
print("===================== Complete===================")
