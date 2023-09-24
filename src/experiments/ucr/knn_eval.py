import argparse
import random

import numpy as np
from sklearn.metrics import accuracy_score

import wandb
from config.config import logger
from src.dp.exact_dp import drop_dtw_distance, dtw_distance
from src.experiments.ucr.utils import get_train_test_data, random_add_noise_with_seed
from src.pow.pow import partial_order_wasserstein
from src.utils.knn_utils import get_distance_matrix_with_ray as get_distance_matrix
from src.utils.knn_utils import knn_classifier_from_distance_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlier_ratio", type=float, default=0.1)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--reg", type=float, default=1)
    parser.add_argument("--distance", type=str, default="euclidean")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main(args):
    wandb.init(project="ucr", entity="sequence-learning", config=args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.info(f"Args: {args}")
    X_train, y_train, X_test, y_test = get_train_test_data(dataset=args.dataset)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    fn_dict = {
        "pow": partial_order_wasserstein,
        "dtw": dtw_distance,
        "drop_dtw": drop_dtw_distance,
    }

    train_size, test_size = X_train.shape[0], X_test.shape[0]
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    X_test_outlier = random_add_noise_with_seed(X_test, args.outlier_ratio, args.seed)
    X_train_outlier = random_add_noise_with_seed(X_train, args.outlier_ratio, args.seed)
    logger.info("X_test_outlier shape: {}".format(X_test_outlier.shape))
    X_test = X_test_outlier
    X_train = X_train_outlier

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
