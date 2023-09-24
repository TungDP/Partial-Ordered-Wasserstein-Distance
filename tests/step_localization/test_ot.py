import torch
import numpy as np
from src.pow.pow import pow_regularization
import pytest


def order_regularization(M,reg):
        I = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                I[i, j] = (i/M.shape[0] - j/M.shape[1])**2
        return M + reg*I



def test_pow_regularization_new(M, reg):

    new_distance = pow_regularization(M, reg)
    old_distance = order_regularization(M, reg)
    assert np.allclose(new_distance, old_distance)
