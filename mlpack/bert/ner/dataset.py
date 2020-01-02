import torch

from mlpack.bert.dataset import FeaturesDataset


class NERDataset(FeaturesDataset):

    def __getitem__(self, idx):
        feat = self.features[idx]
        return torch.tensor(feat.input_ids), torch.tensor(feat.input_mask), \
            torch.tensor(feat.label_ids) - 1, torch.tensor(feat.label_mask)
