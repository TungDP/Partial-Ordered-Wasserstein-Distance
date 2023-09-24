import os

import numpy as np

from config.config import UCR_PATH

UCR_PATH = UCR_PATH / "UCRArchive_2018"


def get_data(path, dataset_size=-1):
    with open(path, "r") as f:
        data = []
        labels = []
        for line in f.readlines():
            label = int(float(line.split()[0]))
            record = [float(j) for j in line.split()[1:]]
            data.append(record)
            labels.append(label)
    if dataset_size > 0:
        data = data[:dataset_size]
        labels = labels[:dataset_size]
    return data, labels


def get_train_test_data(dataset: str):
    train_path = os.path.join(UCR_PATH, dataset, dataset + "_TRAIN.tsv")
    test_path = os.path.join(UCR_PATH, dataset, dataset + "_TEST.tsv")

    X_train, y_train = get_data(train_path)
    X_test, y_test = get_data(test_path, dataset_size=-1)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train, y_test = np.array(y_train, dtype=np.int8), np.array(y_test, dtype=np.int8)
    return X_train, y_train, X_test, y_test


def random_add_noise_with_seed(X, noise_ratio, seed):
    np.random.seed(seed)
    outlier = np.max(X) * np.random.choice([-1, 1], X.shape)
    # choice noise ratio from each row in X without replacement
    outlier_choice = [
        np.random.choice(X.shape[1], int(X.shape[1] * noise_ratio), replace=False)
        for _ in range(X.shape[0])
    ]
    outlier_mask = np.zeros(X.shape)
    for i, choice in enumerate(outlier_choice):
        outlier_mask[i, choice] = 1
    # noise_mask = np.random.choice([0, 1], X.shape, p=[1-noise_ratio, noise_ratio])
    X_noise = X + outlier * outlier_mask
    return X_noise
