import cv2
from matplotlib import pyplot as plt

image0 = cv2.imread('image/ilusi.jpeg')
gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(gray,(3,3),0)
laplacian = cv2.Laplacian(image,cv2.CV_64F)

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original')   
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.xticks([])
plt.yticks([])

plt.show()

img = cv2.imread('image/ilusi.jpeg',0)
blur = cv2.GaussianBlur(img,(3,3),0)

laplacian = cv2.Laplacian(blur, cv2.CV_64F)
laplacian1 = laplacian/laplacian.max()

cv2.imshow('laplacian-gaussian', laplacian1)
cv2.waitKey(0)