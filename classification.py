import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('coins.jpg')
src = cv.imread('coins.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

# sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
_, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv.connectedComponents(sure_fg)

markers += 1

markers[unknown == 255] = 0

markers = cv.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

images = [src, gray, thresh, opening, sure_bg, sure_fg, unknown, markers, image]
titles = ['Original', 'Graubild', 'Schwellenwert', 'Opening', 'Hintergrund', 'Vordergrund', 'Unbekannter Bereich', 'Marker', 'Ergebnis']

for i in range(9):
    if i != 7:
        images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)

    plt.subplot(3, 3, i + 1), plt.imshow(images[i])
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
cv.waitKey(0)

