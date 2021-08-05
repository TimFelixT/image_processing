import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('starfish_1280_960.jpg', 0)
plt.subplot(221), plt.imshow(image, cmap='gray')

_, threshold = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
plt.subplot(222), plt.imshow(threshold, cmap='gray')

image_filtered = cv.Laplacian(threshold, cv.CV_64F)
image_filtered = threshold - image_filtered
image_filtered = np.clip(image_filtered, 0, 255)
plt.subplot(223), plt.imshow(image_filtered, cmap='gray')

plt.show()
