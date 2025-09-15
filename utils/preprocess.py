import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

def load_data(data_path, img_size=(48,48), num_classes=7):
    X, y = [], []
    # Assumes folder structure: data/happy, data/sad, etc.
    for idx, label in enumerate(os.listdir(data_path)):
        folder = os.path.join(data_path, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size) / 255.0
            X.append(img)
            y.append(idx)
    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1)
    y = to_categorical(y, num_classes=num_classes)
    return X, y
