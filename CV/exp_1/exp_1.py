import cv2

img = cv2.imread("/media/galdrux/galdrux_storage/sem_5/CV/exp_1/satellite_9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Original Image", img)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("HSV Image", hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()