import cv2 as cv
import numpy as np

image = cv.imread('starfish_1280_960.jpg', 0)

ret, threshold_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

cv.imshow('threshold', threshold_image)
cv.waitKey(0)