from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split


class WeisDataset:
    def __init__(self, dataset_folder, test_size=0.2):
        self.dataset_folder = dataset_folder
        self.sequence_path = list(Path(dataset_folder).glob("*.pkl"))
        # filename
        labels = set([p.stem.split("_")[1] for p in self.sequence_path])
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.idx2label = {idx: label for idx, label in enumerate(labels)}
        self.train_test_split(test_size=test_size)

    def get_label_name_from_filepath(self, file_path: Path):
        return file_path.stem.split("_")[1]

    @classmethod
    def from_folder(cls, dataset_folder, test_size=0.2):
        return cls(dataset_folder, test_size=test_size)

    def __len__(self):
        return len(self.sequence_path)

    def __getitem__(self, idx):
        return self.get_sequence(idx), self.get_label(idx)

    def get_label(self, idx):
        return self.get_label_name_from_filepath(self.sequence_path[idx])

    def get_sequence(self, idx):
        mask_sequence_flatten = joblib.load(self.sequence_path[idx])
        return mask_sequence_flatten.astype("float64")

    def train_test_split(self, test_size=0.2):
        labels = [self.get_label(idx) for idx in range(len(self))]
        self.train_idx, self.test_idx = train_test_split(
            range(len(self)), test_size=test_size, stratify=labels
        )
