from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import fetch_dataset as fd


if __name__ == '__main__':
    # Download images file
    print("==== Begin download ====")
    data = fd.get_dataset(fd.ds_name, no_of_images=10000)
    fd.set_images(data, dest_path=fd.dest_folder)
    print("==== End download ====")

    print("==== Begin feature extraction ====")
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img/train/images").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)

