from itertools import product

import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch import exp, log

device = "cuda" if torch.cuda.is_available() else "cpu"


def unique_softmax(sim, labels, gamma=1, dim=0):
    assert sim.shape[0] == labels.shape[0]
    labels = labels.detach().cpu().numpy()
    unique_labels, unique_index, unique_inverse_index = np.unique(
        labels, return_index=True, return_inverse=True
    )
    unique_sim = sim[unique_index]
    unique_softmax_sim = torch.nn.functional.softmax(unique_sim / gamma, dim=dim)
    softmax_sim = unique_softmax_sim[unique_inverse_index]
    return softmax_sim


def cosine_sim(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return cos_sim_fn(x[..., None], z.T[None, ...])


def compute_all_costs(
    normal_size_features,
    drop_side_features,
    gamma_xz,
    keep_percentile,
    l2_normalize=False,
    metric="inner",
):
    """This function computes pairwise match and individual drop costs used in Drop-DTW

    Parameters
    __________

    normal_size: torch.tensor of size [N, d]
        step features
    frame_features: torch.tensor of size [M, d]
        frame features
    distractor: torch.tensor of size [d] or None
        Background class prototype. Only used if the drop cost is learnable.
    distractor: torch.tensor of size [d] or None
        Background class prototype. Only used if the drop cost is learnable.
    drop_cost_type: str
        The type of drop cost definition, i.g., learnable or logits percentile.
    keep_percentile: float in [0, 1]
        if drop_cost_type == 'logit', defines drop (keep) cost threshold as logits percentile
    l2_normalize: bool
        wheather to normalize clip and step features before computing the costs
    """
    if l2_normalize:
        frame_features = F.normalize(frame_features, p=2, dim=1)
        step_features = F.normalize(step_features, p=2, dim=1)
    if metric == "inner":
        sim = normal_size_features @ drop_side_features.T
        k = max([1, int(torch.numel(sim) * keep_percentile)])
        baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
        baseline_logits = baseline_logit.repeat([1, sim.shape[1]])
        sims_ext = torch.cat([sim, baseline_logits], dim=0)
        softmax_sims = torch.nn.functional.softmax(sims_ext / gamma_xz, dim=0)
        softmax_sim, drop_probs = softmax_sims[:-1], softmax_sims[-1]
        zx_costs = -torch.log(softmax_sim + 1e-5)
        drop_costs = -torch.log(drop_probs + 1e-5)

        return zx_costs, drop_costs, drop_probs
    else:
        try:
            drop_side_features1 = drop_side_features.numpy()
            normal_size_features1 = normal_size_features.numpy()
            cost = ot.dist(normal_size_features1, drop_side_features1, metric=metric)
        except Exception as e:
            raise e
        # drop_cost = torch.percentile(cost, keep_percentile * 100)
        drop_cost = np.percentile(cost, keep_percentile * 100)
        return (
            torch.tensor(cost),
            torch.repeat_interleave(torch.tensor(drop_cost), cost.shape[1]),
            None,
        )


class VarTable:
    def __init__(self, dims, dtype=torch.float, device=device):
        self.dims = dims
        d1, d2, d_rest = dims[0], dims[1], dims[2:]

        self.vars = []
        for i in range(d1):
            self.vars.append([])
            for j in range(d2):
                var = torch.zeros(d_rest).to(dtype).to(device)
                self.vars[i].append(var)

    def __getitem__(self, pos):
        i, j = pos
        return self.vars[i][j]

    def __setitem__(self, pos, new_val):
        i, j = pos
        if self.vars[i][j].sum() != 0:
            assert (
                False
            ), "This cell has already been assigned. There must be a bug somwhere."
        else:
            self.vars[i][j] = self.vars[i][j] + new_val

    def show(self):
        device, dtype = self[0, 0].device, self[0, 0].dtype
        mat = torch.zeros((self.d1, self.d2, self.d3)).to().to(dtype).to(device)
        for dims in product([range(d) for d in self.dims]):
            i, j, rest = dims[0], dims[1], dims[2:]
            mat[dims] = self[i, j][rest]
        return mat


def minGamma(inputs, gamma=1, keepdim=True):
    """continuous relaxation of min defined in the D3TW paper"""
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minG = inputs.min(dim=0, keepdim=keepdim)
    else:
        # log-sum-exp stabilization trick
        zi = -inputs / gamma
        max_zi = zi.max()
        log_sum_G = max_zi + log(exp(zi - max_zi).sum(dim=0, keepdim=keepdim) + 1e-5)
        minG = -gamma * log_sum_G
    return minG


def minProb(inputs, gamma=1, keepdim=True):
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minP = inputs.min(dim=0, keepdim=keepdim)
    else:
        probs = F.softmax(-inputs / gamma, dim=0)
        minP = (probs * inputs).sum(dim=0, keepdim=keepdim)
    return minP


def traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)
