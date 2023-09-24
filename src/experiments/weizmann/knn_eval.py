import argparse
import random

import numpy as np
from sklearn.metrics import accuracy_score

import wandb
from config.config import WEI_PATH, logger
from src.experiments.weizmann.dataset import WeisDataset
from src.experiments.weizmann.utils import add_outlier
from src.utils.knn_utils import get_distance_matrix_with_ray as get_distance_matrix
from src.utils.knn_utils import knn_classifier_from_distance_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--outlier_ratio", type=float, default=0.1)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--reg", type=float, default=1)
    parser.add_argument("--distance", type=str, default="euclidean")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main(args):
    wandb.init(project="weizmann", entity="sequence-learning", config=args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info(f"Args: {args}")
    weis_dataset = WeisDataset.from_folder(WEI_PATH, test_size=args.test_size)
    X_train = [weis_dataset.get_sequence(idx) for idx in weis_dataset.train_idx]
    X_test = [weis_dataset.get_sequence(idx) for idx in weis_dataset.test_idx]

    X_test = list(
        map(lambda x: add_outlier(x, outlier_ratio=args.outlier_ratio), X_test)
    )
    X_train = list(
        map(lambda x: add_outlier(x, outlier_ratio=args.outlier_ratio), X_train)
    )

    y_train = (weis_dataset.get_label(idx) for idx in weis_dataset.train_idx)
    y_test = (weis_dataset.get_label(idx) for idx in weis_dataset.test_idx)
    y_train = np.array(list(y_train))
    y_test = np.array(list(y_test))

    result = get_distance_matrix(X_train, X_test, args)

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=args.k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy}")
    wandb.run.summary["accuracy"] = accuracy
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
