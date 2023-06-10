import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('image/marvel.jpg',0)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) 
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_prewittx = cv2.filter2D(image, -1, kernelx) 
img_prewitty = cv2.filter2D(image, -1, kernely)

img_prewitt = img_prewittx + img_prewitty

fig, axes = plt.subplots(4, 2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(image, cmap ='gray')
ax[0].set_title("Citra Input")
ax[1].hist(image.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_prewittx, cmap ='gray')
ax[2].set_title("Citra Input Prewitt X")
ax[3].hist(img_prewittx.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Input Prewitt X")

ax[4].imshow(img_prewitty, cmap ='gray')
ax[4].set_title("Citra Input Prewitt Y")
ax[5].hist(img_prewitty.ravel(), bins = 256)
ax[5].set_title("Histogram Citra Input Prewitt Y")

ax[6].imshow(img_prewitt, cmap ='gray')
ax[6].set_title("Citra Input Prewitt")
ax[7].hist(img_prewitt.ravel(), bins = 256)
ax[7].set_title("Histogram Citra Prewitt")

fig.tight_layout()
plt.show()