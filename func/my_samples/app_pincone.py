import pinecone
import pinecone.info
import os
import pickle
import numpy as np
import pandas as pd
from pc_image_embedder import ImageEmbedder
from torchvision import datasets
import random

# Setup Pinecone configuration environment
api_key = os.getenv("PINECONE_API_KEY") or "86e95ed5-3633-498c-b63e-23cc09828cca"
pinecone.init(api_key=api_key, environment="northamerica-northeast1-gcp")

version_info = pinecone.info.version()
server_version = ".".join(version_info.server.split(".")[:2])
client_version = ".".join(version_info.client.split(".")[:2])

IMG_DIR = "/images/train"

# Data preparation
# Copy subset of training image into the project folder
# select_images(train_src_folder, destination_folder+'/train', limit=2000)
# random.seed(0)
# # Select random sample of image classes
# image_classes = set(random.sample(range(200), 5))
# # Get image file names
# image_file_names = [filename for filename, label in datasets.ImageFolder(f"{IMG_DIR}/train").imgs
#                     if label in image_classes]

feature_list = np.array(pickle.load(open('static/index/embeddings_2000.pkl', 'rb')))
filenames = pickle.load(open('static/index/filenames_2000.pkl', 'rb'))

df = pd.DataFrame()
image_embedder = ImageEmbedder()

df["image_filenames"] = filenames
# df['embedding_id'] = [filenames.split(IMG_DIR)[-1] for filename in filenames]
df['embedding'] = [image_embedder.embed(filename) for filename in filenames]
df = df.sample(frac=1)

print(filenames[:2])
print(df.head(2))
