import torch
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        raise NotImplementedError


class NERDataset(FeaturesDataset):

    def __getitem__(self, idx):
        feat = self.features[idx]
        return torch.tensor(feat.input_ids), torch.tensor(feat.input_mask), \
            torch.tensor(feat.label_ids) - 1, torch.tensor(feat.label_mask)
