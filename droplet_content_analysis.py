import pandas as pd
import cv2
import numpy as np
import os
import math


INPUT_FILEPATH = 'droplet_measurements.csv'
OUTPUT_FILEPATH = 'droplet_content_measurements.csv'

IMAGE_DIRECTORY = 'training_data'


def get_file_content() -> pd.DataFrame:
    filepath = INPUT_FILEPATH
    return pd.read_csv(filepath)


def get_luminance(filename, centroid_x, centroid_y) -> float:
    filepath = filename
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    x, y = int(centroid_x), int(centroid_y)      # center coordinate
    r = 10               # half-size of the square region - assumed value. Should consider

    # clamp bounds so they stay inside the image
    h, w = gray.shape
    x1 = max(0, x - r)
    x2 = min(w, x + r + 1)
    y1 = max(0, y - r)
    y2 = min(h, y + r + 1)

    roi = gray[y1:y2, x1:x2]
    avg_brightness = np.mean(roi)

    return avg_brightness


def get_luminance_from_row(row:pd.Series) -> float:
    return get_luminance(row.filename, row.centroid_x, row.centroid_y)


if __name__ == "__main__":
    df = get_file_content()

    df['luminance'] = df.apply(get_luminance_from_row, axis = 1)

    df.to_csv(OUTPUT_FILEPATH)