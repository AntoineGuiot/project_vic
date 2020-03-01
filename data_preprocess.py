import cv2
import numpy as np
import pandas as pd
import skimage

from skimage.feature import hog
from skimage import data, exposure

from sklearn.preprocessing import OneHotEncoder


class dataFormatter:
    def __init__(self, csv):
        self.data = csv
        images = self.data.pixels.values
        for i, image in enumerate(images):
            images[i] = np.array([int(i) for i in self.data.pixels.values[i].split()], dtype=np.uint8).reshape((48, 48))

        self.images = images

    def compute_hog(self):
        hog_features = []
        hog_images = []

        for image in self.images:
            features, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualise=True)
            hog_features.append(features)
            hog_images.append(hog_image)

        self.data['hog_features'] = hog_features

        return hog_images

    def process_target(self):
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        emotion = enc.fit_transform(self.data.emotion.values.reshape(-1, 1))
        for i in range(emotion.shape[1]):
            self.data[enc.categories_[0][i]] = emotion[:, i]
