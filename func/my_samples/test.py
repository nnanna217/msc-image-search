import pickle
import random
import numpy as np
from helper import show_image, show_images_horizontally, select_images
from feature_extractor import FeatureExtractor
from sklearn.neighbors import NearestNeighbors
import cv2

# Load the feature_list and image_filenames
feature_list = np.array(pickle.load(open('static/index/embeddings_50000.pkl', 'rb')))
image_filenames = pickle.load(open('static/index/filenames_50000.pkl', 'rb'))

print(np.array(feature_list).shape)

# Instantiate the FeatureExtractor and extract features for the query image
query_features = FeatureExtractor()
query_image_path = "data/images/sample/sample-4.jpeg"
q = query_features.extract_features(query_image_path)

# Perform k-nearest neighbors search
neighbours = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbours.fit(feature_list)

distances, indices = neighbours.kneighbors(np.array(q).reshape(1, -1))

print(indices)


def display_image(i, index):
    window_names = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5',  'Image 6']
    temp_img = cv2.imread(image_filenames[index])
    cv2.imshow(window_names[i], cv2.resize(temp_img, (256, 256)))


cv2.imshow('Query image', cv2.imread(query_image_path))
query_image_path
for i, file in enumerate(indices[0]):
    display_image(i, file)

    # print(image_filenames[file])
# for i in range(5):
#     show_images_horizontally(random.sample(image_filenames, 10))

cv2.waitKey(0)
cv2.destroyAllWindows()
