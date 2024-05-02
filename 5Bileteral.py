import cv2
import numpy as np
import matplotlib.pyplot as plt
image=cv2.imread('woman.png')
image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
blur1 = cv2.bilateralFilter(image,5,10, 2)
#blur2 = cv2.bilateralFilter(image,10,0, 2)
blur2= cv2.blur(image,(5,5))
plt.figure(figsize=(30,15))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title("original image")
plt.subplot(132)
plt.imshow(blur1, cmap='gray')
plt.subplot(133)
plt.imshow(blur2, cmap='gray')
plt.show()
