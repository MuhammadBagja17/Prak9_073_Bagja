import cv2
from matplotlib import pyplot as plt

image = cv2.imread('image/marvel.jpg',0)
img_canny = cv2.Canny(image,100,200)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(image, cmap ='gray')
ax[0].set_title("Citra Input")
ax[1].hist(image.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_canny, cmap ='gray')
ax[2].set_title("Citra Input Canny")
ax[3].hist(img_canny.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Input Canny")

fig.tight_layout()
plt.show()