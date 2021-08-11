import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('noisy square.jpg', 0)
#global
ret1, threshold1 = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

#otsu
ret2, threshold2 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

#gauss + otsu
blur = cv.GaussianBlur(image, (5, 5), 0)
ret3, threshold3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

images = [image, threshold1, image, threshold2, blur, threshold3]
titles = ['Original', 'Global Thresholding', 'Original', 'Otsu Thresholding', 'Gaussian Filter', 'Gauss + Otsu']

for i in range(6):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
