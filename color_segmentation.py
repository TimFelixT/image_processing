import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random as rng

src = cv.imread(cv.samples.findFile('starfish_1280_960.jpg', 1))
if src is None:
    print('Could not open or find the Image')
    exit(0)

blur = cv.blur(src, (5, 5))
blur1 = cv.GaussianBlur(blur, (5, 5), 0)


hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
low_blue = np.array([0, 55, 0])
high_blue = np.array([255, 250, 255])
mask = cv.inRange(hsv, low_blue, high_blue)

res = cv.bitwise_and(src, src, mask=mask)

# Beseitigung der Störungen
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(res, cv.MORPH_OPEN, kernel, iterations=5)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=5)
cv.imwrite('closing_image.jpg', closing)

# Watershed zum Darstellen der Kanten
gray = cv.cvtColor(closing, cv.COLOR_BGR2GRAY)
sure_bg = cv.dilate(gray, kernel, iterations=10)

dist_transform = cv.distanceTransform(gray, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

ret, markers = cv.connectedComponents(sure_fg)

markers = markers+1
markers[unknown == 255] = 0


markers = cv.watershed(hsv, markers)
hsv[markers == -1] = [0, 255, 255]

# Darstellung
images = [src, blur1, gray, res, hsv]
titles = ['Original', 'Gauß Filter', '-----', 'Color Segmentation', '']

for i in range(len(images)):
    if i != 4:
        images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
    else:
        images[i] = cv.cvtColor(images[i], cv.COLOR_HSV2RGB)
    plt.subplot(3, 3, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()
