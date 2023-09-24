import torch
import torch.nn.functional as F
from src.dp.dp_utils import compute_all_costs as compute_all_costs_new
import numpy as np


def compute_all_costs_them(sample, distractor, gamma_xz, drop_cost_type, keep_percentile, l2_nomalize=False):
    """This function computes pairwise match and individual drop costs used in Drop-DTW

    Parameters
    __________

    sample: dict
        sample dictionary
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

    labels = sample['unique_step_ids']
    step_features, frame_features = sample['unique_step_features'], sample['frame_features']
    if l2_nomalize:
        frame_features = F.normalize(frame_features, p=2, dim=1)
        step_features = F.normalize(step_features, p=2, dim=1)
    sim = step_features @ frame_features.T

    unique_labels, unique_index, unique_inverse_index = np.unique(
        labels.detach().cpu().numpy(), return_index=True, return_inverse=True)
    unique_sim = sim[unique_index]

    if drop_cost_type == 'logit':
        k = max([1, int(torch.numel(unique_sim) * keep_percentile)])
        baseline_logit = torch.topk(unique_sim.reshape([-1]), k).values[-1].detach()
        baseline_logits = baseline_logit.repeat([1, unique_sim.shape[1]])  # making it of shape [1, N]
        sims_ext = torch.cat([unique_sim, baseline_logits], dim=0)
    elif drop_cost_type == 'learn':
        distractor_sim = frame_features @ distractor
        sims_ext = torch.cat([unique_sim, distractor_sim[None, :]], dim=0)
    else:
        assert False, f"No such drop mode {drop_cost_type}"

    unique_softmax_sims = torch.nn.functional.softmax(sims_ext / gamma_xz, dim=0)
    unique_softmax_sim, drop_probs = unique_softmax_sims[:-1], unique_softmax_sims[-1]
    matching_probs = unique_softmax_sim[unique_inverse_index]
    zx_costs = -torch.log(matching_probs + 1e-5)
    drop_costs = -torch.log(drop_probs + 1e-5)
    return zx_costs, drop_costs, drop_probs


def compute_all_costs_our_old(x1, x2, gamma=1, keep_percentile=0.4, l2_nomalize=False):
    sim = x1 @ x2.T
    k = int(np.prod(sim.shape) * keep_percentile)
    baseline_logit = np.sort(sim.reshape([-1]))[-k]
    sims_ext = np.zeros([sim.shape[0]+1, sim.shape[1]])
    sims_ext[:-1, :] = sim
    sims_ext[-1, :] = baseline_logit

    softmax_sims = np.exp(sims_ext / gamma) / np.sum(np.exp(sims_ext / gamma))

    def softmax_numpy(x, axis=None):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    softmax_sims = softmax_numpy(sims_ext / gamma, axis=0)
    softmax_sim, drop_probs = softmax_sims[:-1], softmax_sims[-1]
    zx_costs = -np.log(softmax_sim + 1e-5)
    x2_drop_costs = -np.log(drop_probs + 1e-5)
    return zx_costs, x2_drop_costs


def test_inner_distance(sample):
    #khong model -> gamma = 1 co model gamma = 30
    distractor = None
    gamma_xz = 1
    drop_cost_type = 'logit'
    keep_percentile = 0.3
    distance = 'inner'

    zx_costs, drop_costs = compute_all_costs_our_old(x1=sample['unique_step_features'], x2=sample['frame_features'], gamma=gamma_xz, keep_percentile=keep_percentile, l2_nomalize=False)

    zx_costs_them, drop_costs_them, drop_probs_them = compute_all_costs_them(sample=sample, distractor=distractor, gamma_xz=gamma_xz, drop_cost_type=drop_cost_type, keep_percentile=keep_percentile, l2_nomalize=False)
    zx_costs_them, drop_costs_them = zx_costs_them.detach().cpu().numpy(), drop_costs_them.detach().cpu().numpy()


    zx_costs_new, drop_costs_new, drop_probs_new = compute_all_costs_new(normal_size_features=sample['unique_step_features'], drop_side_features=sample['frame_features'], gamma_xz=gamma_xz, keep_percentile=keep_percentile, l2_normalize=False, metric=distance)
    zx_costs_new, drop_costs_new = zx_costs_new.detach().cpu().numpy(), drop_costs_new.detach().cpu().numpy()

    assert zx_costs.shape == (sample['unique_step_features'].shape[0], sample['frame_features'].shape[0])
    assert drop_costs.shape == (sample['frame_features'].shape[0],)

    assert np.allclose(zx_costs, zx_costs_new, atol=1e-2)
    assert np.allclose(drop_costs, drop_costs_new, atol=1e-2)

    assert np.allclose(zx_costs_new, zx_costs_them)
    assert np.allclose(drop_costs_new, drop_costs_them)
