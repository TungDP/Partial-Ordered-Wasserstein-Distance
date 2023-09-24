import os
from glob import glob
from os import path as osp

import numpy as np
import pytorch_lightning as pl
import torch

from config.config import COIN_PATH, CT_PATH, YC_PATH, logger
from src.experiments.step_localization.data.data_utils import dict2tensor
from src.experiments.step_localization.data.loader import (
    LMDB_Class_Dataset,
    LMDB_Folder_Dataset,
)

from .data_module import DataModule


class Unique_LMDB_Class_Dataset(LMDB_Class_Dataset):
    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        _, inx = np.unique(sample_dict["step_ids"], return_index=True)
        sorted_inx = np.sort(inx)
        unique_step_ids = sample_dict["step_ids"][sorted_inx]
        sample_dict["unique_step_ids"] = unique_step_ids
        sample_dict["unique_step_features"] = sample_dict["step_features"][np.sort(inx)]
        sample_dict["num_unique_steps"] = torch.tensor(
            len(sample_dict["unique_step_ids"])
        )
        sample_dict["normal_to_unique"] = torch.tensor(
            [unique_step_ids.tolist().index(s) for s in sample_dict["step_ids"]]
        )
        self.add_gt_assignment(sample_dict)
        return sample_dict

    @staticmethod
    def add_gt_assignment(sample):
        num_frames = sample["num_frames"].numpy().squeeze()
        gt_assignment = -torch.ones(num_frames, dtype=torch.int32)
        normal_to_unique = sample["normal_to_unique"].numpy().squeeze()
        if len(normal_to_unique.shape) == 0:
            normal_to_unique = np.expand_dims(normal_to_unique, axis=0)
        for idx, s in enumerate(normal_to_unique):
            st_ed = torch.arange(
                sample["step_starts"][idx], sample["step_ends"][idx] + 1
            )
            gt_assignment[st_ed] = s
        sample["gt_assignment"] = gt_assignment


class Unique_LMDB_Folder_Dataset(LMDB_Folder_Dataset):
    def __init__(self, folder, split="train", transform=None, truncate=0):
        cls_folders = []
        for cls_folder in glob(osp.join(folder, "*/")):
            files = glob(osp.join(cls_folder, "*.lmdb"))
            file_has_split = ["_{}".format(split) in f for f in files]
            if any(file_has_split):
                cls_folders.append(cls_folder)

        # instantiating datasets for each class
        self.cls_datasets = [
            Unique_LMDB_Class_Dataset(f, split, transform, truncate)
            for f in cls_folders
        ]
        self.cls_lens = [len(d) for d in self.cls_datasets]
        self.cls_end_idx = np.cumsum(self.cls_lens)


class UniqueDataModule(DataModule):
    def __init__(self, dataset_name, n_cls, batch_size):
        pl.LightningDataModule.__init__(self)
        if dataset_name == "COIN":
            folder = COIN_PATH
        elif dataset_name == "YouCook2":
            folder = YC_PATH
        elif dataset_name == "CrossTask":
            folder = CT_PATH
        else:
            raise f"No such dataset {dataset_name}"

        self.lmdb_path = os.path.join(folder, "lmdb")
        self.n_cls = n_cls
        self.batch_size = batch_size

        self.train_dataset = Unique_LMDB_Folder_Dataset(
            self.lmdb_path, split="train", transform=dict2tensor
        )
        self.val_dataset = Unique_LMDB_Folder_Dataset(
            self.lmdb_path, split="val", transform=dict2tensor
        )
        self.test_dataset = Unique_LMDB_Folder_Dataset(
            self.lmdb_path, split="test", transform=dict2tensor
        )
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
