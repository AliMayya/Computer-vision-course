import cv2
import numpy as np
import matplotlib.pyplot as plt
image=cv2.imread("edges.jpg",0)
Hy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
horizontal_filtering= cv2.filter2D(src=image, kernel=Hy,ddepth=-1)

# Sobel
Hx= np.array( [[1,0,-1],[2,0,-2],[1,0,-1]])
vertical_filtering=cv2.filter2D(src=image,kernel=Hx, ddepth=-1)
final_image=np.sqrt(pow(horizontal_filtering,2.0)+ 
pow(vertical_filtering, 2.0))
plt.figure(figsize=(30,15))
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title("original image")
plt.subplot(222)
plt.imshow(horizontal_filtering, cmap='gray')
plt.title("image with horizantal sobel filtering")
plt.subplot(223)
plt.imshow(vertical_filtering, cmap='gray')
plt.title("image with vertical sobel filtering")
plt.subplot(224)
plt.imshow(final_image, cmap='gray')
plt.title("Sobel Image Gradient")
plt.show()

# Laplcian
image=cv2.imread('edge.jpg',0)
laplacian1=np.array([[0,1,0],[1,-4,1],[0,1,0]])
laplacian2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
laplacian3=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
laplacian4=np.array([[-1,-1,-1],[-1,+8,-1],[-1,-1,-1]])
image_laplacian1=cv2.filter2D(src=image, kernel=laplacian1, ddepth=-1)
image_laplacian2=cv2.filter2D(src=image, kernel=laplacian2, ddepth=-1)
image_laplacian3=cv2.filter2D(src=image, kernel=laplacian3, ddepth=-1)
image_laplacian4=cv2.filter2D(src=image, kernel=laplacian4, ddepth=-1)
plt.figure(figsize=(30,15))
plt.subplot(231)
plt.imshow(image, cmap='gray')
plt.title("original image")
plt.subplot(232)
plt.imshow(image_laplacian1, cmap='gray')
plt.title("[0,1,0],[1,-4,1],[0,1,0]")
plt.subplot(233)
plt.imshow(image_laplacian2, cmap='gray')
plt.title("[1,1,1],[1,-8,1],[1,1,1]")
plt.subplot(234)
plt.imshow(image_laplacian3, cmap='gray')
plt.title("[0,-1,0],[-1,4,-1],[0,-1,0]")
plt.subplot(235)
plt.imshow(image_laplacian4, cmap='gray')
plt.title("[-1,-1,-1],[-1,+8,-1],[-1,-1,-1]")
plt.show()

