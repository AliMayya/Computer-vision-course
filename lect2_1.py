import numpy as np
import cv2
from matplotlib import pyplot as plt
### gray and binary transformation
a=cv2.imread("rose.jpg",1)
gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',gray)
cv2.waitKey()

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('Binary Image', binary)
###
# Convert the grayscale image to double precision

double_gray = gray.astype(np.float64)
double_gray=double_gray/255
double_gray = double_gray**0.5
print(np.max(double_gray))
print(np.min(double_gray))
fig, axs = plt.subplots(2, 1)

# Show the original image
axs[0].imshow(gray, cmap='gray')
axs[1].imshow(double_gray, cmap='gray')
plt.show()



