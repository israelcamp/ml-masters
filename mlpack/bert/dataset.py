import torch
from torch.utils.data import Dataset


class InputFeaturesCollection:

    def __init__(self, features):
        self.features = sorted(
            features, key=lambda x: int(x.unique_id.split('_')[-1]))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    @property
    def batch(self):
        input_ids, input_masks, labels = [], [], []
        for feature in self.features:
            input_ids.append(feature.input_ids)
            input_masks.append(feature.input_mask)
            labels.append(feature.label_ids)
        input_ids = torch.tensor(input_ids)
        input_masks = torch.tensor(input_masks)
        labels = torch.tensor(labels)
        return input_ids, input_masks, labels


class InputFeaturesCollectionExtended(InputFeaturesCollection):

    @property
    def batch(self):
        input_ids, input_masks, label_ids, label_mask, max_context = [], [], [], [], []
        for feature in self.features:
            input_ids.append(feature.input_ids)
            input_masks.append(feature.input_mask)
            label_ids.append(feature.label_ids)
            label_mask.append(feature.label_mask)

            # max context
            tok_context = []
            tok_context.append(0)  # for [CLS]
            for i in range(len(feature.token_is_max_context)):
                tok_context.append(feature.token_is_max_context[i+1] * 1)
            tok_context.append(0)  # for the [SEP]
            while len(tok_context) < 512:
                tok_context.append(0)
            max_context.append(tok_context)

        input_ids = torch.tensor(input_ids).long()
        input_masks = torch.tensor(input_masks).long()
        label_ids = torch.tensor(label_ids)
        label_mask = torch.tensor(label_mask).long()
        max_context = torch.tensor(max_context)
        return input_ids, input_masks, label_ids, label_mask, max_context


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
