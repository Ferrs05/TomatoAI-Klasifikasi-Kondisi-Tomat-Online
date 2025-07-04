import os
import cv2
import numpy as np
import tensorflow as tf

LABELS = ['matang', 'belum_matang', 'rusak']
IMAGE_SIZE = (128, 128)

def load_data_cnn(data_dir):
    """
    Muat gambar dari subfolder LABELS di data_dir,
    kembalikan X (normalized) dan y (one-hot).
    """
    X, y = [], []
    for idx, label in enumerate(LABELS):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            X.append(img)
            y.append(idx)
    X = np.array(X, dtype='float32') / 255.0
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=len(LABELS))
    return X, y