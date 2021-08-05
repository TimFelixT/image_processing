import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread('starfish_1280_960.jpg', 0)

kernel = np.ones((3, 3), np.uint8)

image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=8)
image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=8)

edges = cv.Canny(image, 30, 200)

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
