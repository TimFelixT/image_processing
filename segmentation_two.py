import cv2 as cv
import numpy as np
import random as rng

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
cv.imshow('New Sharped Image', image_Result)


bw = cv.cvtColor(image_Result, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cv.imshow('Binary Image', bw)

dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
cv.imshow('Distance Transform Image', dist)

_, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)

kernel1 = np.ones((3, 3), dtype=np.uint8)
dist = cv.dilate(dist, kernel1)
cv.imshow('Peaks', dist)

dist_8u = dist.astype('uint8')

_, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

markers = np.zeros(dist.shape, dtype=np.int32)

for i in range(len(contours)):
    cv.drawContours(markers, contours, i, (i+1), -1)

cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
markers_8u = (markers * 10).astype('uint8')
cv.imshow('Markers', markers_8u)

# Watershed Algorithm
cv.watershed(image_Result, markers)

mark = markers.astype('uint8')
mark = cv.bitwise_not(mark)

colors = []
for contour in contours:
    colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))

dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i, j]
        if index > 0 and index <= len(contours):
            dst[i, j, :] = colors[index-1]

cv.imshow('Final Result', dst)

cv.waitKey(0)
