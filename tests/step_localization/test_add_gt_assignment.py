
import numpy as np
from copy import deepcopy
import torch

from src.experiments.step_localization.data.unique_data_module import Unique_LMDB_Class_Dataset

def test_add_gt_assignment_true(sample):
    before = deepcopy(sample)
    Unique_LMDB_Class_Dataset.add_gt_assignment(sample)
    assert torch.all(sample['gt_assignment'] == before['gt_assignment'])

def test_add_gt_assignment_false(sample_wrong):
    before = deepcopy(sample_wrong)
    Unique_LMDB_Class_Dataset.add_gt_assignment(sample_wrong)
    assert not torch.all(sample_wrong['gt_assignment'] == before['gt_assignment'])
