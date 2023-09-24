import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from config.config import COIN_PATH, CT_PATH, YC_PATH, logger
from src.experiments.step_localization.data.batching import (
    BatchIdxSampler_Class,
    flatten_batch,
)
from src.experiments.step_localization.data.data_utils import dict2tensor
from src.experiments.step_localization.data.loader import LMDB_Folder_Dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, n_cls, batch_size):
        super().__init__()
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

        self.train_dataset = LMDB_Folder_Dataset(
            self.lmdb_path, split="train", transform=dict2tensor
        )
        self.val_dataset = LMDB_Folder_Dataset(
            self.lmdb_path, split="val", transform=dict2tensor
        )
        self.test_dataset = LMDB_Folder_Dataset(
            self.lmdb_path, split="test", transform=dict2tensor
        )
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Val dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        batch_idx_sampler = BatchIdxSampler_Class(
            self.train_dataset, self.n_cls, self.batch_size
        )
        train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_idx_sampler,
            collate_fn=flatten_batch,
            num_workers=32,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, collate_fn=flatten_batch, num_workers=32
        )
        return val_loader

    def test_dataloader(self):
        val_loader = DataLoader(
            self.test_dataset, collate_fn=flatten_batch, num_workers=32
        )
        return val_loader
