import pytest
import torch
import numpy as np


@pytest.fixture
def gt_assignment():
    return [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1,  0,  0,  1,  1,  2,  2,  2,  2, -1, -1, -1,  3,  3,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  1,  1, -1, -1,  3,  3,  3,  3,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  3,  3,  3, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

@pytest.fixture
def other_assignment():
    return [-1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,  2,  3, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


@pytest.fixture
def other_assignment2():
    return [-1, -1, -1,  0,  0,  0, -1,  0, -1, -1, -1,  0,  1, -1,  0, -1, -1,
       -1, -1, -1, -1, -1, -1,  0,  0,  0, -1, -1,  0,  0, -1, -1, -1, -1,
       -1, -1, -1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1,  1,  1,
        2,  1, -1,  1,  1,  2, -1, -1, -1, -1, -1, -1, -1,  2, -1, -1, -1,
       -1,  2, -1, -1, -1, -1,  3,  3, -1, -1, -1, -1,  3, -1, -1, -1,  3,
        3, -1,  3, -1, -1, -1, -1, -1,  2, -1, -1,  2,  2,  3,  3,  3,  3,
        2,  2,  2, -1, -1, -1,  3, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


@pytest.fixture
def sample(gt_assignment):
    sample = torch.load('tests/data/sample.pt')
    sample['gt_assignment'] = torch.tensor(gt_assignment)
    return sample

@pytest.fixture
def sample_wrong(other_assignment):
    sample = torch.load('tests/data/sample.pt')
    sample['gt_assignment'] = torch.tensor(other_assignment)
    return sample
    # return {
    #     'name': 'UseVolumetricFlask_173_vigsBzBWcCk',
    #     'cls': torch.tensor(173),
    #     'cls_name': 'UseVolumetricFlask',
    #     'num_subs': torch.tensor(1),
    #     'num_steps': torch.tensor(6),
    #     'num_frames': torch.tensor(151),
    #     'step_ids': torch.tensor([281, 282, 279, 280, 282, 280]),
    #     'step_starts_sec': torch.tensor([223., 230., 235., 257., 292., 305.]),
    #     'step_starts': torch.tensor([69, 71, 73, 80, 91, 95]),
    #     'step_ends_sec': torch.tensor([227., 234., 245., 291., 295., 355.]),
    #     'step_ends': torch.tensor([ 70,  73,  76,  90,  92, 110]),
    #     'unique_step_ids': torch.tensor([281, 282, 279, 280]),
    #     'num_unique_steps': 4,
    #     'normal_to_unique': torch.tensor([0, 1, 2, 3, 1, 3]),
    #     'gt_assignment': torch.tensor(other_assignment),
    # }


@pytest.fixture
def M():
    # return torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    # numpy
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

@pytest.fixture
def reg():
    return 0.1

@pytest.fixture
def m():
    return 0.8
