import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from mlpack.utils import save_model

try:
    from apex import amp
except ImportError:
    print(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


class TrainArgs:
    def __init__(self, num_epochs=10, ckp_path='model_checkpoint.ckp'):
        self.num_epochs = num_epochs
        self.ckp_path = ckp_path


class BaseTrainer:

    def __init__(self,
                 grad_steps: int = 1,
                 device=None,
                 fp16: bool = False,
                 max_grad_norm: float = 1.,
                 writer=None,
                 notebook=True
                 ):
        self.grad_steps = grad_steps
        self.fp16 = fp16
        self.max_grad_norm = max_grad_norm
        self.writer = writer

        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_iter, self.epoch = 0, 0
        self.best_metric = 0.
        if notebook:
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm

        self.tqdm = tqdm

    @staticmethod
    def dataloader_generator(dataloader):
        '''
        DataLoader generator, should return a dictionary of tensors
        '''
        raise NotImplementedError

    @staticmethod
    def loss_from_model(model_output, targets, loss_fn=None):
        raise NotImplementedError

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
                o = model(**inputs)

            loss = self.loss_from_model(o, targets, loss_fn)

            y = targets['y']

            preds += o.argmax(1).detach().cpu().numpy().tolist()
            trues += y.detach().cpu().numpy().tolist()
            losses.append(loss.item())

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds)
        conf = confusion_matrix(trues, preds)

        print('--- Validation ---')
        print(f'F1 = {f1}\t Acc = {acc}')
        print(conf)
        return np.array(losses).mean(), f1

    def train(self, args: TrainArgs, model, dl_train, dl_valid, optimizer, loss_fn=None,
              scheduler=None):

        device = self.device

        for ep in self.tqdm(range(args.num_epochs), desc='Training...'):
            model.train()
            losses_train = []
            for step, batch in self.tqdm(enumerate(self.dataloader_generator(dl_train)), leave=False, total=len(dl_train)):
                inputs = batch['inputs']
                targets = batch['targets']
                output = model(**inputs)

                loss = self.loss_from_model(output, targets, loss_fn)

                losses_train.append(loss.item())

                if self.grad_steps > 1:
                    loss = loss / self.grad_steps

                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm)

                if (step + 1) % self.grad_steps == 0 or (step + 1) == len(dl_train):
                    optimizer.step()
                    model.zero_grad()
                    if scheduler:
                        scheduler.step()

                if self.writer:
                    self.writer.add_scalar('loss/train', loss, self.n_iter)
                self.n_iter += 1

            print(
                f'-- Train Loss {sum(losses_train)/len(losses_train)}', flush=True)

            # evaluate
            valid_loss, valid_metric = self.evaluate_fn(
                model, dl_valid, loss_fn)

            if self.writer:
                self.writer.add_scalar('loss/valid', valid_loss, self.epoch)
                self.writer.add_scalar(
                    'metric/valid', valid_metric, self.epoch)
                for name, param in model.named_parameters():
                    self.writer.add_histogram(name, param, self.epoch)

            self.epoch += ep

            print(
                f'---Valid\nLoss {valid_loss}\nMetric {valid_metric}', flush=True)

            if valid_metric > self.best_metric:
                self.best_metric = valid_metric
                save_model(model, optimizer, args.ckp_path,
                           scheduler=scheduler)
