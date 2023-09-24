import numpy as np
import ot


def relative_element_trans(x, metric="euclidean"):
    if metric == "euclidean":
        x_tru = np.concatenate((np.zeros((1, x.shape[1])), x[:-1]), axis=0)
        distance = np.linalg.norm(x - x_tru, axis=1)
        fx = np.cumsum(distance)
        sum_distance = fx[-1]
        fx = fx / sum_distance
        return fx


def t_opw1(X, Y, a=None, b=None, lambda1=50, lambda2=0.1, delta=1, metric="euclidean"):
    """t preserved OT 1

    Args:
        X (ndarray): view1
        Y (ndarray): view2
        lambda1 (int, optional): weight of first term. Defaults to 50.
        lambda2 (float, optional): weight of second term. Defaults to 0.1.
        delta (int, optional): _description_. Defaults to 1.

    Returns:
        distance, ot_plan: distance is the distance between views, ot_plan is the transport plan
    """
    tolerance = 0.5e-2
    maxIter = 20

    N = X.shape[0]
    M = Y.shape[0]
    dim = X.shape[1]
    if dim != Y.shape[1]:
        print("X and Y must have the same number of columns")
        return

    if a is None:
        a = np.ones((N, 1)) / N
    if b is None:
        b = np.ones((M, 1)) / M

    # mid_para = np.sqrt((1 / (N ** 2) + 1 / (M ** 2)))

    row_col_matrix = np.mgrid[1 : N + 1, 1 : M + 1]
    row = row_col_matrix[0] / N  # row = (i+1)/N
    col = row_col_matrix[1] / M  # col = (j+1)/M

    # d_matrix = np.abs(row - col) / mid_para
    fx = relative_element_trans(X)
    fy = relative_element_trans(Y)
    m, n = np.meshgrid(fy, fx)
    d_matrix = np.maximum(m, n) / np.minimum(m, n)

    P = np.exp(-(d_matrix**2) / (2 * delta**2)) / (delta * np.sqrt(2 * np.pi))
    P = a @ b.T * P

    S = lambda1 / ((row - col) ** 2 + 1)  # S = lamda1 * E in paper

    D = ot.dist(X, Y, metric=metric)
    D = D / np.max(D)

    # Clip the distance matrix to prevent numerical errors
    # max_distance = 200 * lambda2
    # D = np.clip(D, 0, max_distance)
    K = np.exp((S - D) / lambda2) * P

    # a = np.ones((N, 1)) / N
    # b = np.ones((M, 1)) / M

    compt = 0
    u = np.ones((N, 1)) / N

    while compt < maxIter:
        u = a / (K @ (b / (K.T @ u)))
        assert not np.isnan(u).any(), "nan in u"
        compt += 1

        if compt % 20 == 0 or compt == maxIter:
            v = b / (K.T @ u)
            u = a / (K @ v)

            criterion = np.linalg.norm(
                np.sum(np.abs(v * (K.T @ u) - b), axis=0), ord=np.inf
            )
            if criterion < tolerance:
                break

    U = K * D
    dis = np.sum(u * (U @ v))
    T = np.diag(u[:, 0]) @ K @ np.diag(v[:, 0])
    # dis = np.sum(T * D)
    return dis
