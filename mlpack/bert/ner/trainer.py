import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from seqeval.metrics import classification_report

from mlpack.trainer import BaseTrainer
from mlpack.utils import to_device
from mlpack.bert.ner.eval import f1_per_label


class BertNERTrainer(BaseTrainer):

    def __init__(self, *args, ner_labels, **kwargs):
        super().__init__(*args, **kwargs)
        self.ner_labels = ner_labels

    def dataloader_generator(self, dataloader):
        for input_ids, input_mask, label_ids, label_mask in dataloader:
            input_ids, input_mask, label_ids, label_mask = to_device(input_ids, input_mask, label_ids,
                                                                     label_mask, device=self.device)
            d = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'label_ids': label_ids,
                'label_mask': label_mask
            }
            yield {
                'inputs': d,
                'targets': d
            }

    @staticmethod
    def loss_from_model(model_output, targets, loss_fn=None):
        return model_output[0]

    def evaluate_fn(self, model, dataloader, loss_fn):
        model.eval()
        losses = []
        preds = []
        trues = []
        dl_gen = self.dataloader_generator(dataloader)
        for batch in self.tqdm(dl_gen, leave=False, desc='Eval...', total=len(dataloader)):
            inputs = batch['inputs']
            targets = batch['targets']

            with torch.no_grad():
                loss, active_logits, active_labels = model(**inputs)

            losses.append(loss.item())

            active_logits = active_logits.argmax(dim=1).cpu().numpy()
            active_labels = active_labels.cpu().numpy()
            accs = (1 * (active_logits == active_labels)).tolist()

            # transforming
            ts = [
                self.ner_labels[y] for y in active_labels
            ]
            ps = [
                self.ner_labels[y] for y in active_logits
            ]
            preds += ps
            trues += ts

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='micro', labels=self.ner_labels)
        conf = confusion_matrix(trues, preds, labels=self.ner_labels)
        f1_on_labels = f1_per_label(trues, preds)

        s = '--- Validation ---'
        s += f'\nF1 = {f1}\t Acc = {acc}'
        s += f'\nF1 per Label {f1_on_labels}'
        s += f'\nClass. Report:\n{classification_report(trues, preds)}'
        s += f'\nConfusion Matrix:\n{conf}'
        self.train_logger.info(s)
        return np.array(losses).mean(), f1
