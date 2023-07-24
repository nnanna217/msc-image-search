import pickle
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Perform read operations and extract image features
feature = FeatureExtractor()
# features = []
# img_paths = []

# Load the feature_list and image_filenames
features = np.array(pickle.load(open('static/index/embeddings_2000.pkl', 'rb')))
image_paths = pickle.load(open('static/index/filenames_2000.pkl', 'rb'))

# for feature_path in Path("./static/feature").glob("*.npy"):
#     features.append(np.load(feature_path))
#     img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query_img"]

        # save and load image to be queried
        img = Image.open(file.stream)
        upload_img_path = f"static/uploads/" + datetime.now().isoformat().replace(":", "." + file.filename)
        print(upload_img_path)
        img.save(upload_img_path)

        # perform search
        # query = feature.extract_features(img)
        # dists = np.linalg.norm(features - query, axis=1)
        # ids = np.argsort(dists)[:30]
        # scores =  [(dists[id], img_paths[id]) for id in ids]

        # print(scores)
        return render_template("index.html", query_path=upload_img_path)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()
