import os
import numpy as np
from sklearn.model_selection import train_test_split


def image_shape():
    return 30, 40


def load_data(test_size=0.16, random_state=42):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orl_data_path = os.path.join(data_dir, 'orl_data.txt')
    orl_labels_path = os.path.join(data_dir, 'orl_lbls.txt')
    X = np.loadtxt(orl_data_path).transpose()
    y = np.loadtxt(orl_labels_path, dtype=np.uint8)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
