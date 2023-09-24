import numpy as np
import ot
import scipy
from ot.backend import get_backend
from ot.optim import line_search_armijo
from ot.utils import list_to_array


def pow_regularization(M, reg):
    I = get_I(M)
    return M + reg * I


def get_I(M):
    rows, cols = M.shape
    i, j = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing="ij"
    )  # Use np.meshgrid instead of torch.meshgrid
    I = ((i / rows - j / cols) ** 2).astype(M.dtype)
    return I


def get_assignment(soft_assignment):
    """Get assignment from soft assignment"""
    assignment = np.argmax(soft_assignment, axis=0)
    outlier_label = soft_assignment.shape[0] - 1
    assignment[assignment == outlier_label] = -1
    return assignment


def partial_order_wasserstein(
    M,
    order_reg,
    m,
    p=None,
    q=None,
    nb_dummies=1,
    ot_algo="emd",
    sinkhorn_reg=0.1,
    return_dist=True,
    **kwargs
):
    """Solves the partial optimal transport problem
    and returns the OT plan

    Parameters
    ----------
    M : ndarray, shape (ns, nt)
        Cost matrix
    p : ndarray, shape (ns,), optional
        Masses in the source domain
    q : ndarray, shape (nt,), optional
        Masses in the target domain
    m : float, optional
        Total mass to be transported from source to target
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    order_reg : float
        Order regularization parameter
    nb_dummies : int
        Number of dummy points to add (avoid instabilities in the EMD solver)
    ot_algo : str, optional
        OT solver to use (default: "emd")  either "emd" or "sinkhorn"
    sinkhorn_reg : float, optional
        Sinkhorn regularization parameter (default: 0.1) if ot_algo="sinkhorn"
    return_dist : bool, optional
        If True, returns the partial order wasserstein distance (default: False) else returns the OT plan

    Returns
    -------
    T : ndarray, shape (ns, nt) or float if return_dist=True
    """
    if p is None:
        p = np.ones(M.shape[0]) / M.shape[0]
    if q is None:
        q = np.ones(M.shape[1]) / M.shape[1]

    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater" " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    dim_M_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    M_reg = pow_regularization(M, order_reg)

    M_emd = np.zeros(dim_M_extended, dtype=M.dtype)
    M_emd[: len(p), : len(q)] = M_reg
    M_emd[-nb_dummies:, -nb_dummies:] = np.max(M) * 1e2
    if ot_algo == "emd":
        T, logemd = ot.emd(p_extended, q_extended, M_emd, log=True, **kwargs)
    elif ot_algo == "sinkhorn":
        T = ot.sinkhorn(p_extended, q_extended, M_emd, reg=sinkhorn_reg, log=False)

    if return_dist:
        return np.sum(T[: len(p), : len(q)] * M)
    else:
        return T[: len(p), : len(q)]


def partial_order_wasserstein_for_step_localization(
    M,
    order_reg,
    p=None,
    q=None,
    m=None,
    nb_dummies=1,
    ot_algo="emd",
    sinkhorn_reg=0.1,
    return_outliers=True,
    return_dist=False,
    **kwargs
):
    """Solves the partial optimal transport problem for step localization ( 1side extension)
    and returns the OT plan (with dummy points as outliers if return_outliers=True)

    Parameters
    ----------
    M : ndarray, shape (ns, nt)
        Cost matrix
    p : ndarray, shape (ns,), optional
        Masses in the step side
    q : ndarray, shape (nt,), optional
        Masses in the frame side
    m : float, optional
        Total mass to be transported from step to frame
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    order_reg : float
        Order regularization parameter
    nb_dummies : int
        Number of dummy points to add (avoid instabilities in the EMD solver)
    ot_algo : str, optional
        OT solver to use (default: "emd")  either "emd" or "sinkhorn"
    sinkhorn_reg : float, optional
        Sinkhorn regularization parameter (default: 0.1) if ot_algo="sinkhorn"

    Returns
    -------
    T : ndarray, shape (ns+1, nt)
    """

    if p is None:
        p = np.ones(M.shape[0]) / M.shape[0]
    if q is None:
        q = np.ones(M.shape[1]) / M.shape[1]

    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater" " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    p = p * m

    dim_M_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    M_reg = pow_regularization(M, order_reg)

    M_emd = np.zeros(dim_M_extended, dtype=M.dtype)
    M_emd[: len(p), : len(q)] = M_reg
    M_emd[-nb_dummies:, -nb_dummies:] = np.max(M) * 1e2
    # M_emd[: len(p), -nb_dummies:] = np.max(M) * 1e2
    if ot_algo == "emd":
        T, logemd = ot.emd(p_extended, q_extended, M_emd, log=True, **kwargs)
    elif ot_algo == "sinkhorn":
        T = ot.sinkhorn(p_extended, q_extended, M_emd, reg=sinkhorn_reg, log=False)

    if return_dist:
        return np.sum(T[: len(p), : len(q)] * M)
    elif return_outliers:
        drop_part = np.sum(T[-nb_dummies:, : len(q)], axis=0)
        transfer_part = np.vstack((T[: len(p), : len(q)], drop_part))
        return transfer_part
    else:
        return T[: len(p), : len(q)]


def generic_conditional_gradient(
    a,
    b,
    M,
    f,
    df,
    reg1,
    reg2,
    lp_solver,
    line_search,
    G0=None,
    numItermax=200,
    stopThr=1e-9,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    **kwargs
):
    a, b, M, G0 = list_to_array(a, b, M, G0)
    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(a, b)
    elif a is None or b is None:
        nx = get_backend(M)
    else:
        nx = get_backend(a, b, M)
    loop = 1

    if log:
        log = {"loss": []}

    if G0 is None:
        # G = nx.outer(a, b)
        m = kwargs.get("m", None)
        G = get_I(nx.zeros((a.shape[0], b.shape[0])))
        G = G / G.sum() * m
    else:
        # to not change G0 in place.
        G = nx.copy(G0)

    if reg2 is None:

        def cost(G):
            return nx.sum(M * G) + reg1 * f(G)

    else:

        def cost(G):
            # return nx.sum(M * G) + reg1 * f(G) + reg2 * nx.sum(G * nx.log(G))
            return nx.sum(M * G) + reg1 * f(G) + reg2 * nx.sum(get_I(G) * G)

    cost_G = cost(G)
    if log:
        log["loss"].append(cost_G)

    it = 0

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}|{:8s}".format(
                "It.", "Loss", "Relative loss", "Absolute loss"
            )
            + "\n"
            + "-" * 48
        )
        print("{:5d}|{:8e}|{:8e}|{:8e}".format(it, cost_G, 0, 0))

    while loop:
        it += 1
        old_cost_G = cost_G
        # problem linearization
        Mi = M + reg1 * df(G)

        if not (reg2 is None):
            Mi = Mi + reg2 * get_I(G)
        # set M positive
        Mi = Mi + nx.min(Mi)  # FIXME: check if this is needed

        # solve linear program
        if log:
            Gc, innerlog_ = lp_solver(a, b, Mi, **kwargs)
        else:
            Gc = lp_solver(a, b, Mi, **kwargs)

        # line search
        deltaG = Gc - G

        alpha, fc, cost_G = line_search(cost, G, deltaG, Mi, cost_G, **kwargs)

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_cost_G = abs(cost_G - old_cost_G)
        relative_delta_cost_G = abs_delta_cost_G / abs(cost_G)
        if relative_delta_cost_G < stopThr or abs_delta_cost_G < stopThr2:
            loop = 0

        if log:
            log["loss"].append(cost_G)

        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}|{:8s}".format(
                        "It.", "Loss", "Relative loss", "Absolute loss"
                    )
                    + "\n"
                    + "-" * 48
                )
            print(
                "{:5d}|{:8e}|{:8e}|{:8e}".format(
                    it, cost_G, relative_delta_cost_G, abs_delta_cost_G
                )
            )

    if log:
        log.update(innerlog_)
        return G, log
    else:
        return G


def partial_order_wasserstein_with_reg(
    a,
    b,
    M,
    order_reg,
    smooth_reg,
    f,
    df,
    G0=None,
    numItermax=10,
    stopThr=1e-9,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    **kwargs
):
    def lp_solver(a, b, Mi, **kwargs):
        return partial_order_wasserstein_for_step_localization(
            M=Mi,
            order_reg=0,  # add order reg outside inner loop
            m=kwargs.get("m"),
            return_outliers=False,
            nb_dummies=kwargs.get("nb_dummies", 1),
        )

    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        kwargs.pop("m")
        return line_search_armijo(cost, G, deltaG, Mi, cost_G, **kwargs)

    tmp = generic_conditional_gradient(
        a,
        b,
        M,
        f,
        df,
        smooth_reg,
        order_reg,
        lp_solver,
        line_search,
        G0=G0,
        numItermax=numItermax,
        stopThr=stopThr,
        stopThr2=stopThr2,
        verbose=verbose,
        log=log,
        **kwargs
    )
    if log:
        G, _ = tmp
    else:
        G = tmp

    # Restore the drop part of G
    def get_full_matrix(T):
        drop_line = 1 / T.shape[1] - np.sum(T, axis=0)
        return np.vstack([T, drop_line])

    G = get_full_matrix(G)

    return G


def step_localization(
    M,
    order_reg,
    smooth_reg,
    m,
    numItermax=10,
    stopInnerThr=1e-9,
    verbose=False,
    log=False,
):
    """Solve the step localization problem with the given regularization parameters.

    Parameters
    ----------
    M : ndarray, shape (ns, nt)
        Cost matrix.
    order_reg : float
        Order regularization parameter.
    smooth_reg : float
        Smoothness regularization parameter.
    m : float
        Total mass to be transported from source to target.

    Returns
    -------
    T : ndarray, shape (ns + 1, nt)


    """

    def get_D(T):
        N, M = T.shape
        ones_row = np.ones((1, M))
        return scipy.sparse.spdiags(
            np.vstack((ones_row, -2 * ones_row, ones_row)), range(3), M - 2, M
        )

    def f(T):
        D = get_D(T)
        return np.linalg.norm(D @ T.T) ** 2

    def df(T):
        D = get_D(T)

        dfT = np.zeros(T.shape, dtype=np.float64)
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                J = np.zeros(T.shape)
                J[i, j] = 1
                dfT[i, j] = 2 * np.trace(D.T @ D @ T.T @ J)
        return dfT

    a = np.ones(M.shape[0]) / M.shape[0]
    b = np.ones(M.shape[1]) / M.shape[1]
    return partial_order_wasserstein_with_reg(
        a,
        b,
        M,
        order_reg,
        smooth_reg,
        f,
        df,
        m=m,
        numItermax=numItermax,
        stopThr=stopInnerThr,
        verbose=verbose,
        log=log,
    )
