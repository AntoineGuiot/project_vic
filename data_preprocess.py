import cv2
import numpy as np
import pandas as pd
import skimage
import dlib

from skimage.feature import hog, local_binary_pattern
from skimage import data, exposure

from sklearn.preprocessing import OneHotEncoder

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


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

    ## compute landmarks features

    def compute_landmarks(self):
        landmarks = []
        for image in self.images:
            face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
            face_landmarks = np.matrix([[p.x, p.y] for p in predictor(image, face_rects[0]).parts()])
            landmarks.append(face_landmarks)
        self.data['landmarks'] = landmarks
        return landmarks

    def compute_lbp(self):
        radius = 1
        n_points = 8 * radius
        sub_region_size = 16
        n_bins = 255
        lbp_hist_list = []
        for image in self.images:
            lbp_hist = []
            lbp = local_binary_pattern(image, n_points, radius, 'default')
            splited_lbp = lbp.reshape(48 // sub_region_size, sub_region_size, -1, 8).swapaxes(1, 2).reshape(-1,
                                                                                                            sub_region_size,
                                                                                                            sub_region_size)
            for region in splited_lbp:
                hist, _ = np.histogram(region, density=True, bins=n_bins, range=(0, n_bins))
                lbp_hist.append(hist)

            lbp_hist = np.concatenate(lbp_hist)

            lbp_hist_list.append(lbp_hist)
        self.data['lbp'] = lbp_hist_list
        self.data['is_lbp_nan'] = np.sum(np.stack(self.data.lbp.values), axis=1)
        return lbp_hist_list

    def process_target(self):
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        emotion = enc.fit_transform(self.data.emotion.values.reshape(-1, 1))
        for i in range(emotion.shape[1]):
            self.data[enc.categories_[0][i]] = emotion[:, i]
