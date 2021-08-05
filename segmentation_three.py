import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread("demonstration-image.png")

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 100, 0.2)

k = 3
_, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
labels = labels.flatten()

segmented_image = centers[labels.flatten()]

segmented_image = segmented_image.reshape(image.shape)

plt.imshow(segmented_image)
plt.show()

masked_image = np.copy(image)

masked_image = masked_image.reshape((-1, 3))

cluster = 2
masked_image[labels == cluster] = [0, 0, 0]

masked_image = masked_image.reshape(image.shape)

plt.imshow(masked_image)
plt.show()
