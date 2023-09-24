import pytest
from src.experiments.step_localization.metrics import framewise_accuracy, IoU
import numpy as np
import torch

class TestFramewiseAccuracy:
    def test_convert_torch(self, gt_assignment, other_assignment):
        gt_assignment = torch.tensor(gt_assignment)
        other_assignment = torch.tensor(other_assignment)
        assert framewise_accuracy(other_assignment, gt_assignment, use_unlabeled=False) == 0.10810810810810811
        assert framewise_accuracy(other_assignment, gt_assignment, use_unlabeled=True) == 0.5298013245033113

    def test_numpy(self, gt_assignment, other_assignment2):
        other_assignment2 = np.array(other_assignment2)
        gt_assignment = np.array(gt_assignment)
        assert framewise_accuracy(other_assignment2, gt_assignment, use_unlabeled=False) == 0.24324324324324326
        assert framewise_accuracy(other_assignment2, gt_assignment, use_unlabeled=True) == 0.6357615894039735

class TestIoU:
    def test_1(self):
        gt = np.array([1,2,3,0,-1])
        other = np.array([1,2,3,0,4])
        assert IoU(other, gt) == 1

    def test_0(self):
        gt = np.array([1,2,3,0,-1])
        other = np.array([2,3,0,1,0])
        assert IoU(other, gt) == 0
