import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from PIL import Image


def download_dataset():
    return


def select_images(source_folder, destination_folder, limit=2000):
    image_count = 0

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Empty the destination folder if it already exists
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)

    # Recreate the destination folder
    os.makedirs(destination_folder, exist_ok=True)

    # Traverse through the files in the source folder
    for file in tqdm(os.listdir(source_folder)):
        # Check if the file is an image (you can modify the condition based on your image file extensions)
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)

            # Copy the image file to the destination folder
            shutil.copyfile(source_path, destination_path)

            image_count += 1

            # Check if the limit has been reached
            if image_count >= limit:
                return

    print(f"Total {image_count} images copied to {destination_folder}")


def show_images_horizontally(filenames):
    n = len(filenames)
    fig, ax = plt.subplots(1, n)
    fig.set_figwidth(1.5 * n)
    for a, f in zip(ax, filenames):
        a.imshow(Image.open(f))
        a.axis("off")
    plt.show()


def show_image(filename):
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(1.3)
    ax.imshow(Image.open(filename))
    ax.axis("off")
