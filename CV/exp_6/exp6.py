import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/media/galdrux/galdrux_storage/sem_5/CV/dataset/satellite_9.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the image path.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

low_contrast = cv2.normalize(gray, None, 90, 160, cv2.NORM_MINMAX)

histogram = cv2.calcHist([low_contrast], [0], None, [256], [0, 256])
normalized_histogram = histogram / low_contrast.size
cdf = normalized_histogram.cumsum()

min_val = np.min(low_contrast)
max_val = np.max(low_contrast)
    
contrast_stretched = ((low_contrast - min_val) / (max_val - min_val) * 255).astype(np.uint8)

equalized = cv2.equalizeHist(low_contrast)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(low_contrast)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ycrcb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
color_equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

images = {
    "Input": low_contrast,
    "Contrast Stretching": contrast_stretched,
    "Global Equalization": equalized,
    "CLAHE": clahe_image
}

for name, img in images.items():
    print("\n" + name)
    print("Minimum:", np.min(img))
    print("Maximum:", np.max(img))
    print("Mean:", np.mean(img))
    print("Standard Deviation:", np.std(img))

print("\nNormalized Histogram Sum:", normalized_histogram.sum())

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(low_contrast, cmap="gray")
plt.title("Low Contrast Input")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(contrast_stretched, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(equalized, cmap="gray")
plt.title("Global Histogram Equalization")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(clahe_image, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(color_equalized)
plt.title("Y Channel Equalization")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(low_contrast.ravel(), 256, [0, 256])
plt.title("Input Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 2)
plt.hist(contrast_stretched.ravel(), 256, [0, 256])
plt.title("Contrast Stretched Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 3)
plt.hist(equalized.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2, 2, 4)
plt.hist(clahe_image.ravel(), 256, [0, 256])
plt.title("CLAHE Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(normalized_histogram, label="Normalized Histogram")
plt.plot(cdf, label="CDF")
plt.title("Normalized Histogram and CDF")
plt.xlabel("Pixel Intensity")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()