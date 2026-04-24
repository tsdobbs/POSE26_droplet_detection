import cv2
import numpy as np

img = cv2.imread("Scan10005.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x, y = 400, 300      # center coordinate
r = 10               # half-size of the square region

# clamp bounds so they stay inside the image
h, w = gray.shape
x1 = max(0, x - r)
x2 = min(w, x + r + 1)
y1 = max(0, y - r)
y2 = min(h, y + r + 1)

roi = gray[y1:y2, x1:x2]
avg_brightness = np.mean(roi)

print("Average brightness:", avg_brightness)
