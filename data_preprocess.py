import cv2
import numpy as np
import pandas as pd


class dataFormatter:
    def __init__(self, csv):
        self.csv = csv
        images = self.csv.pixels.values
        for i, image in enumerate(images):
            images[i] = np.array([int(i) for i in self.csv.pixels.values[i].split()], dtype=np.uint8).reshape((48, 48))

        self.images = images

    def compute_hog(self):
        hist = []
        # image = cv2.imread("test.jpg",0)
        winSize = (48, 48)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)

        for image in self.images:
            hist.append(hog.compute(image))

        return hist

