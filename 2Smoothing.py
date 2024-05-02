import cv2
from matplotlib import pyplot as plt
import numpy as np
image=cv2.imread("Panda.jpg")
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gauss=np.random.normal(0,0.5,gray_image.shape).astype('uint8')
gauss_image=cv2.add(gray_image,gauss)
mean_filter_image=cv2.blur(gray_image,(5,5))
# Create a single subplot for all images
fig, axs = plt.subplots(1, 3)
axs[0].imshow(gray_image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(gauss_image, cmap='gray')
axs[1].set_title('noisy Image')
axs[2].imshow(mean_filter_image, cmap='gray')
axs[2].set_title('Smoothed Image')
plt.show()

gray_image=cv2.imread('salt and pepper noise.png',0)
Median_filter_image=cv2.medianBlur(gray_image,3)
# Create a single subplot for all images
fig, axs = plt.subplots(1, 2)
axs[0].imshow(gray_image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(Median_filter_image, cmap='gray')
axs[1].set_title('Smoothed Image')
plt.show()


gray_image=cv2.imread('space.png',0)
#h=np.matrix([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
k = (1/(11*11))*np.ones((11,11)) 
Mean_filter_image=cv2.filter2D(src=gray_image, kernel=k, ddepth=-1)
_,bw_image=cv2.threshold(Mean_filter_image, 127, 255, cv2.THRESH_BINARY)
# Create a single subplot for all images
fig, axs = plt.subplots(1, 3)
axs[0].imshow(gray_image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(Mean_filter_image, cmap='gray')
axs[1].set_title('Smoothed Image')
axs[2].imshow(bw_image, cmap='gray')
axs[2].set_title('Segmented Image')
plt.show()
