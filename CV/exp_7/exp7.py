import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/media/galdrux/galdrux_storage/sem_5/CV/dataset/satellite_9.jpg"

img = cv2.imread(image_path, 0)

if img is None:
    print("Image not found. Check the image path.")
    exit()

threshold = 127
binary = np.where(img > threshold, 1, 0).astype(np.uint8)

kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.uint8)

def erosion(image, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            if np.all(region[kernel == 1] == 1):
                result[i, j] = 1

    return result

def dilation(image, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            if np.any(region[kernel == 1] == 1):
                result[i, j] = 1

    return result

erosion_result = erosion(binary, kernel)

dilation_result = dilation(binary, kernel)

opening_result = dilation(erosion_result, kernel)

closing_result = erosion(dilation_result, kernel)

images = [
    binary,
    erosion_result,
    dilation_result,
    opening_result,
    closing_result
]

titles = [
    "Binary Image",
    "Erosion",
    "Dilation",
    "Opening",
    "Closing"
]

plt.figure(figsize=(12, 8))

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Numerical Analysis")
print("------------------")

for title, image in zip(titles, images):
    foreground = np.sum(image == 1)
    background = np.sum(image == 0)
    total = image.size
    percentage = (foreground / total) * 100

    print(f"\n{title}")
    print(f"Foreground pixels: {foreground}")
    print(f"Background pixels: {background}")
    print(f"Foreground percentage: {percentage:.2f}%")