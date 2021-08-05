import cv2 as cv
# [variables]
# Declare the variables we are going to use
ddepth = cv.CV_16S
kernel_size = 3
window_name = "Laplace Demo"
# [variables]
# [load]

src = cv.imread(cv.samples.findFile('starfish_1280_960.jpg'), cv.IMREAD_COLOR) # Load an image
# Check if image is loaded fine
# [load]
# [reduce_noise]
# Remove noise by blurring with a Gaussian filter
src = cv.GaussianBlur(src, (3, 3), 0)
# [reduce_noise]
# [convert_to_gray]
# Convert the image to grayscale
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# [convert_to_gray]
# Create Window
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
# [laplacian]
# Apply Laplace function
dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
# [laplacian]
# [convert]
# converting back to uint8
abs_dst = cv.convertScaleAbs(dst)
# [convert]
# [display]
cv.imshow(window_name, abs_dst)
cv.waitKey(0)
# [display]