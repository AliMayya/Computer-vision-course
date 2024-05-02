###
import numpy as np
import cv2
from matplotlib import pyplot as plt
# Load the image
img = cv2.imread('rose.jpg',0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Apply CLAHE to the image
equ = clahe.apply(img)

# Calculate the histogram of the original and CLAHE image
hist = cv2.calcHist([img],[0],None,[256],[0,255])
hist_equ = cv2.calcHist([equ],[0],None,[256],[0,255])

# Create subplots
fig, axs = plt.subplots(2, 2)

# Show the original image
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')

# Show the histogram of the original image
axs[0, 1].plot(hist)
axs[0, 1].set_title('Histogram of Original Image')

# Show the CLAHE image
axs[1, 0].imshow(equ, cmap='gray')
axs[1, 0].set_title('CLAHE Image')

# Show the histogram of the CLAHE image
axs[1, 1].plot(hist_equ)
axs[1, 1].set_title('Histogram of CLAHE Image')

# Display the plot
plt.show()
