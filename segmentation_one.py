import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random as rng

src = cv.imread(cv.samples.findFile('starfish_1280_960.jpg', 1))
if src is None:
    print('Could not open or find the Image')
    exit(0)

gray_image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, threshold = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
low_blue = np.array([55, 0, 0])
high_blue = np.array([118, 255, 255])
mask = cv.inRange(hsv, low_blue, high_blue)

kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(hsv, cv.MORPH_OPEN, kernel, iterations=3)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=60)
cv.imwrite('closing_image.jpg', closing)

sure_bg = cv.dilate(closing, kernel, iterations=10)

dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

ret, markers = cv.connectedComponents(sure_fg)

markers = markers+1
markers[unknown == 255] = 0


markers = cv.watershed(src, markers)
src[markers == -1] = [255, 255, 0]

src = cv.cvtColor(src, cv.COLOR_BGR2RGB)

plt.subplot(2, 1, 1), plt.imshow(src, cmap='gray')
plt.title('Watershed'), plt.xticks([]), plt.yticks([])

plt.show()
