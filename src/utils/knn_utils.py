import numpy as np
import ot
import ray
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from config.config import logger
from src.baselines.opw import opw_discrepancy
from src.baselines.softdtw import soft_dtw_discrepancy
from src.baselines.topw1 import t_opw1
from src.dp.exact_dp import drop_dtw_distance, dtw_distance
from src.pow.pow import partial_order_wasserstein


def knn_classifier_from_distance_matrix(distance_matrix, k, labels):
    """
    Computes the k-nearest neighbors for each point in a dataset given a precomputed distance matrix
    and returns the predicted class labels based on the majority class of its k-nearest neighbors.

    Parameters:
    -----------
    distance_matrix: array-like or sparse matrix, shape (n_test_samples, n_train_samples)
        The precomputed distance matrix.
    k: int
        The number of neighbors to use for classification.
    labels: array-like, shape (n_train_samples,)
        The class labels for each data point in the dataset.

    Returns:
    --------
    predicted_labels: array-like, shape (n_samples,)
        The predicted class labels for each point in the dataset.
    """
    if np.min(distance_matrix) < 0:
        distance_matrix = distance_matrix - np.min(distance_matrix)
    knn_clf = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", metric="precomputed"
    )
    n_train_samples = distance_matrix.shape[1]
    knn_clf.fit(np.random.rand(n_train_samples, n_train_samples), labels)
    predicted_labels = knn_clf.predict(distance_matrix)
    return predicted_labels


fn_dict = {
    "pow": partial_order_wasserstein,
    "dtw": dtw_distance,
    "drop_dtw": drop_dtw_distance,
    "topw1": t_opw1,
    "opw": opw_discrepancy,
    "softdtw": soft_dtw_discrepancy,
}

fn_ray_dict = {k: ray.remote(fn_dict[k]) for k in fn_dict}


def get_distance_matrix(X_train, X_test, args):
    train_size = len(X_train)
    test_size = len(X_test)
    logger.info(f"Train size: {train_size}")
    logger.info(f"Test size: {test_size}")

    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            x_tr = X_train[train_idx].reshape(X_train[train_idx].shape[0], -1)
            x_te = X_test[test_idx].reshape(X_test[test_idx].shape[0], -1)
            M = ot.dist(x_tr, x_te, metric=args.distance)
            if args.metric == "pow":
                distance = fn_dict[args.metric](M=M, order_reg=args.reg, m=args.m)
            elif args.metric == "drop_dtw":
                distance = fn_dict[args.metric](M, keep_percentile=args.m)
            elif args.metric == "topw1":
                distance = fn_dict[args.metric](
                    X=X_train[train_idx], Y=X_test[test_idx], metric=args.distance
                )
            else:
                distance = fn_dict[args.metric](M)
            result[test_idx, train_idx] = distance
    return result


def get_distance_matrix_with_ray(X_train, X_test, args):
    fn_dict = fn_ray_dict
    train_size = len(X_train)
    test_size = len(X_test)
    logger.info(f"Train size: {train_size}")
    logger.info(f"Test size: {test_size}")
    ray.init()
    # result = np.zeros((test_size, train_size))
    result = []
    for test_idx in tqdm(range(test_size)):
        for train_idx in tqdm(range(train_size), leave=False):
            x_tr = X_train[train_idx].reshape(X_train[train_idx].shape[0], -1)
            x_te = X_test[test_idx].reshape(X_test[test_idx].shape[0], -1)
            M = ot.dist(x_tr, x_te, metric=args.distance)
            if args.metric == "pow":
                distance = fn_dict[args.metric].remote(
                    M, m=args.m, order_reg=args.reg, return_dist=True
                )
            elif args.metric == "drop_dtw":
                distance = fn_dict[args.metric].remote(M, keep_percentile=args.m)
            elif args.metric == "topw1":
                distance = fn_dict[args.metric].remote(
                    X=X_train[train_idx], Y=X_test[test_idx], metric=args.distance
                )
            else:
                distance = fn_dict[args.metric].remote(M)
            # elif args.metric == "dtw":
            #     distance = fn_dict[args.metric](M)

            # elif args.metric == "pow":
            #     distance = fn_dict[args.metric](M)
            # elif args.metric == "softdtw":
            #     distance = fn_dict[args.metric](M)
            result.append(distance)

    result = ray.get(result)
    result = np.array(result).reshape(test_size, train_size)
    ray.shutdown()
    return result
