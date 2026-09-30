import cv2
import numpy as np

img = cv2.imread("/media/galdrux/galdrux_storage/sem_5/CV/exp_2/satellite_9.jpg")

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gaussian = cv2.GaussianBlur(gray, (7, 7), 0)

median = cv2.medianBlur(gray, 7)

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

edges = cv2.Canny(gray, 50, 150)

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Gaussian Blur", gaussian)
cv2.imshow("Median Blur", median)
cv2.imshow("Sharpened", sharpened)
cv2.imshow("Canny Edge Detection", edges)

cv2.imwrite("gray.jpg", gray)
cv2.imwrite("gaussian.jpg", gaussian)
cv2.imwrite("median.jpg", median)
cv2.imwrite("sharpened.jpg", sharpened)
cv2.imwrite("edges.jpg", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()