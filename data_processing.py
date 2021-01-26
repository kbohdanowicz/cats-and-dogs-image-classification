from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import pickle
from image_conversion import image_to_pixels, image_to_color_histogram
from utils import measure_time
from pathlib import Path


DATASET_DIRECTORY = 'dogs_vs_cats_25000/'


@measure_time
def get_images_data(load_from_file: bool) -> tuple:
    print('[INFO]: getting images')
    if load_from_file:
        with open("images_data/raw_images.txt", "rb") as file_path:
            raw_images = (pickle.load(file_path))

        with open("images_data/histogram.txt", "rb") as file_path:
            histogram = pickle.load(file_path)

        with open("images_data/labels.txt", "rb") as file_path:
            labels = pickle.load(file_path)
    else:
        raw_images = []
        histogram = []
        labels = []

        image_paths = list(Path(f'{DATASET_DIRECTORY}').rglob("*.jpg"))

        for (i, image_path) in enumerate(image_paths):
            image_path = str(image_path)
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-1].split(".")[0]

            pixels = image_to_pixels(image)
            temp_histogram = image_to_color_histogram(image)

            raw_images.append(pixels)
            histogram.append(temp_histogram)
            labels.append(label)

            if i > 0 and i % 1000 == 0:
                msg = f"[INFO]: processed {i}/{len(image_paths)}"
                print('\r', msg, end='')
            elif i == len(image_paths) - 1:
                msg = f"[INFO]: processed {i + 1}/{len(image_paths)}"
                print('\r', msg, end='')
        print()

        with open("images_data/raw_images.txt", "wb") as file_path:
            pickle.dump(raw_images, file_path)

        with open("images_data/histogram.txt", "wb") as file_path:
            pickle.dump(histogram, file_path)

        with open("images_data/labels.txt", "wb") as file_path:
            pickle.dump(labels, file_path)

        np_arr_raw_images = np.array(raw_images)
        np_arr_histogram = np.array(histogram)

        print("[INFO] raw images matrix: {:.2f}MB".format(np_arr_raw_images.nbytes / (1024 * 1000.0)))
        print("[INFO] histogram matrix: {:.2f}MB".format(np_arr_histogram.nbytes / (1024 * 1000.0)))

    return raw_images, histogram, labels


def get_train_test_data(converted_images: tuple, labels: tuple, ratio: float) -> tuple:
    return train_test_split(converted_images, labels, test_size=ratio, random_state=42)
