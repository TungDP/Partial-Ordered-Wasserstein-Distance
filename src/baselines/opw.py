import math

import numpy as np
import ot

from src.baselines.utils import get_E_F


def opw_discrepancy(M, lambda1=50, lambda2=0.1, delta=1):
    rows, cols = M.shape
    a = np.ones(rows, dtype=M.dtype) / rows
    b = np.ones(cols, dtype=M.dtype) / cols
    E, F = get_E_F(a.shape[0], b.shape[0], backend=np)
    M_hat = (
        M
        - lambda1 * E
        + lambda2 * (F / (2 * delta**2) + np.log(delta * np.sqrt(2 * math.pi)))
    )
    # return ot.sinkhorn2(a, b, M_hat, reg=lambda2)
    return ot.emd2(a, b, M_hat)
