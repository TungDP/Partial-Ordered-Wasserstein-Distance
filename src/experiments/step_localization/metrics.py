import numpy as np
import torch


def framewise_accuracy(frame_assignment, gt_assignment, use_unlabeled=False):
    "Calculate framewise accuracy"
    # convert data type

    if torch.is_tensor(frame_assignment):
        frame_assignment = frame_assignment.detach().cpu().numpy()
    if torch.is_tensor(gt_assignment):
        gt_assignment = gt_assignment.detach().cpu().numpy()
    num_frames = frame_assignment.shape[0]
    if not use_unlabeled:
        unlabled = np.count_nonzero(gt_assignment == -1)
        num_frames = num_frames - unlabled
        fa = np.logical_and(
            frame_assignment == gt_assignment, gt_assignment != -1
        ).sum()
    else:
        fa = np.count_nonzero(frame_assignment == gt_assignment)
    fa = fa / num_frames if num_frames != 0 else 0
    return fa


def IoU(frame_assignment, gt_assignment):
    if torch.is_tensor(frame_assignment):
        frame_assignment = frame_assignment.detach().cpu().numpy()
    if torch.is_tensor(gt_assignment):
        gt_assignment = gt_assignment.detach().cpu().numpy()
    num_unique_steps = np.max(gt_assignment) + 1
    frame_assignment.shape[0]

    intersection, union = 0, 0
    for s in range(num_unique_steps):
        intersection += np.logical_and(gt_assignment == s, frame_assignment == s).sum()
        union += np.logical_or(gt_assignment == s, frame_assignment == s).sum()
    return intersection / union
