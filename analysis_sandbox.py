# This file is a collection of sandbox scripts for trying to analyze droplet images.
# Nothing works well yet!


########################################################################
# Use HoughCircles method to detect circles and output an image displaying these circles.
# Works okay at detecting droplets, but is probably not the right approach.
########################################################################

import cv2 as cv
import numpy as np

image_path = "./images/Scan10006.jpg"
output_path = "./output/Scan10006.png"

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(
    img, cv.HOUGH_GRADIENT, 1, 300, param1=100, param2=30, minRadius=10, maxRadius=400
)
assert circles is not None, "No circles detected"

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # img, (x,y), r, color, ??
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imwrite(output_path, cimg)


########################################################################
# Detect contours and write data to an image file and csv with area, circularity,
# and position data
########################################################################

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path

input_path = "./images/Scan10006.jpg"
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Could not read {input_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

_, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = img.copy()
rows = []

i = 1
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 20:
        continue

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
    cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
    cv2.putText(
        vis,
        str(i),
        (cx + 5, cy - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    rows.append(
        {
            "id": i,
            "area": area,
            "circularity": circularity,
            "centroid_x": cx,
            "centroid_y": cy,
        }
    )
    i += 1

out_img = output_dir / "contours.png"
out_csv = output_dir / "contours.csv"

cv2.imwrite(str(out_img), vis)
pd.DataFrame(rows).to_csv(out_csv, index=False)

with open(str(out_img) + ".meta.json", "w") as f:
    json.dump(
        {
            "caption": "Detected droplet contours",
            "description": "Contour outlines overlaid on the original image, with each droplet labeled.",
        },
        f,
    )

print(f"Saved: {out_img}")
print(f"Saved: {out_csv}")
