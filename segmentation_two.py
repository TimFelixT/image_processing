import cv2 as cv
import numpy as np
import random as rng
from matplotlib import pyplot as plt

rng.seed(12345)


src = cv.imread(cv.samples.findFile('starfish_1280_960.jpg', 1))
if src is None:
    print('Could not open or find the Image')
    exit(0)

kernel = np.ones((3, 3), np.uint8)

src = cv.morphologyEx(src, cv.MORPH_CLOSE, kernel, iterations=5)
src = cv.morphologyEx(src, cv.MORPH_OPEN, kernel, iterations=5)

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

image_Laplacian = cv.filter2D(src, cv.CV_32F, kernel)

sharp = np.float32(src)
image_Result = sharp - image_Laplacian

image_Result = np.clip(image_Result, 0, 255)
image_Result = image_Result.astype('uint8')
image_Laplacian = np.clip(image_Laplacian, 0, 255)
image_Laplacian = np.uint8(image_Laplacian)

plt.subplot(221), plt.imshow(image_Result, cmap='gray')
plt.title('Laplace Operator'), plt.xticks([]), plt.yticks([])

bw = cv.cvtColor(image_Result, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

plt.subplot(222), plt.imshow(bw, cmap='gray')
plt.title('Schwellenwert Binarisierung'), plt.xticks([]), plt.yticks([])

dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)

plt.subplot(223), plt.imshow(dist, cmap='gray')
plt.title('Distanz Transoformation'), plt.xticks([]), plt.yticks([])

_, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)

kernel1 = np.ones((3, 3), dtype=np.uint8)
dist = cv.dilate(dist, kernel1)

plt.subplot(224), plt.imshow(dist, cmap='gray')
plt.title('Dilation'), plt.xticks([]), plt.yticks([])

plt.show()
