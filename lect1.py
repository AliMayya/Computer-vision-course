import numpy as np
import cv2
from matplotlib import pyplot as plt
a=cv2.imread("rose.jpg")
cv2.imshow('image',a)
cv2.waitKey()
g=cv2.imread("rose.jpg",0)
cv2.imshow('image',g)
cv2.waitKey()
cv2.imwrite('R.tiff',a, [cv2.IMWRITE_JPEG_QUALITY, 100])

print(np.max(a))
print(np.min(a))
print(a.shape)
print(a.dtype)
print(a.size)

print(np.iinfo(a.dtype))

###

a=cv2.imread(r"C:\Users\Acer\Desktop\Computer Vision 2023\practical\1\Chapter 1\lecture1\rose.jpg")
b=a.astype(float)
print(b.dtype)
print(np.max(b))
print(np.min(b))




