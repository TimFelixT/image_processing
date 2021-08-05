import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('coins.jpg')
gray = cv.cvtColor(image, cv.Color_BGR2GRAY)
re, thresh = cv.threshold(gray, 0 ,255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)


cv.imshow('Output', image)
cv.waitKey(0)

