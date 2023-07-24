from pathlib import Path
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

# Define the dataset name
ds_name = "SaffalPoosh/deepFashion-with-masks"

# Load the dataset
# dataset = load_dataset(dataset_name)
# Save the dataset to a local directory
# dataset.save_to_disk("static/img/")
dest_folder = Path("./static/img/train/images")
test_folder = Path("./static/img/test")


def get_dataset(dataset_name=ds_name, no_of_images=10000):
    dataset = load_dataset(dataset_name, split=f"train[:{no_of_images}]")
    return dataset


def get_test_images(dataset, idx):
    dataset = load_dataset(dataset, split=f"train[{idx+1}:{idx+21}]")
    return dataset


def get_images(dataset, key='images'):
    images = []
    for n in range(0, len(dataset)):
        images.append(np.array(dataset[n][key]))
    return images


def set_images(dataset, dest_path):
    for n in range(0, len(dataset)):
        image = np.array(dataset[n]['images'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{dest_path}/{dataset[n]['pid']}.jpg", image)


def display_images(dataset):
    fig, axes = plt.subplots(5, 5, figsize=(5,5))
    axes = axes.flatten()

    for i in range(25):
        image = dataset[i]
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def delete_files_with_extension(directory, extension, exclude_files):
    file_pattern = os.path.join(directory, f"*.{extension}")
    files = glob.glob(file_pattern)

    for file in files:
        if file not in exclude_files:
            os.remove(file)
            print(f"Deleted file: {file}")


# directory_to_search = dest_folder
# extension_to_delete = "jpg"
# files_to_exclude = ["images3.jpg", "sample2.jpg", "images_new.jpg", "output.jpg"]
#
# delete_files_with_extension(directory_to_search, extension_to_delete, files_to_exclude)



# print(dest_folder)

images = get_test_images(ds_name, 10000)
set_images(images, test_folder)
print('end')
