import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from mlpack.utils import create_dir

try:
    from apex import amp
except ImportError:
    print(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


class TrainArgs:
    def __init__(self, num_epochs=10, ckp_name='model_checkpoint.ckp'):
        self.num_epochs = num_epochs
        self.ckp_name = ckp_name

    def __str__(self):
        return f'Epochs: {self.num_epochs} - Ckp Name: {self.ckp_name}'


class BaseTrainer:

    def __init__(self,
                 grad_steps: int = 1,
                 device=None,
                 fp16: bool = False,
                 max_grad_norm: float = 1.,
                 log_dir='model/',
                 notebook=True,
                 ):
        self.grad_steps = grad_steps
        self.fp16 = fp16
        self.max_grad_norm = max_grad_norm
        self.log_dir = log_dir
        self.notebook = notebook
        create_dir(log_dir)
        self.train_logger = self._init_train_logger()
        self.params_logger = self._init_params_logger()
        self.writer = SummaryWriter(self.tb_dir)

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

    @property
    def train_file(self):
        return os.path.join(self.log_dir, 'train.log')

    @property
    def tb_dir(self):
        return os.path.join(self.log_dir, 'tensorboard/')

    @property
    def params_file(self):
        return os.path.join(self.log_dir, 'params.log')

    @staticmethod
    def dataloader_generator(dataloader):
        '''
        DataLoader generator, should return a dictionary of tensors
        >>> for x, y in dataloader:
        >>>   yield {
        >>>     'inputs':{
        >>>        'input': x   
        >>>      },
        >>>      'targets': {
        >>>          'y': y
        >>>      }
        >>>   }
        '''
        raise NotImplementedError

    @staticmethod
    def loss_from_model(model_output, targets, loss_fn=None):
        raise NotImplementedError

    @staticmethod
    def _optimizer_ckp_path(ckp_path):
        fmt = ckp_path.split('/')[-1].split('.')[-1]
        optim_path = ckp_path.replace(f'.{fmt}', f'_optimizer.{fmt}')
        return optim_path

    @staticmethod
    def _scheduler_ckp_path(ckp_path):
        fmt = ckp_path.split('/')[-1].split('.')[-1]
        sched_path = ckp_path.replace(f'.{fmt}', f'_lrscheduler.{fmt}')
        return sched_path

    def _init_logger(self, name, handlers):
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        for h in handlers:
            h.setFormatter(formatter)
            logger.addHandler(h)
        return logger

    def _init_train_logger(self):
        handlers = [logging.FileHandler(self.train_file)]
        if not self.notebook:
            handlers.append(logging.StreamHandler())
        return self._init_logger('train', handlers)

    def _init_params_logger(self):
        return self._init_logger('params', [
            logging.FileHandler(self.params_file)
        ])

    def logtrain_string(self, s):
        if self.notebook:
            print(s, flush=True)
        self.train_logger.info(s)

    def model_checkpoint_path(self, ckp_name):
        return os.path.join(self.log_dir, ckp_name)

    def save_model(self, model, optimizer, ckp_name, scheduler=None):
        ckp_path = self.model_checkpoint_path(ckp_name)
        torch.save(model.state_dict(), ckp_path)
        # saving the optimizer
        optim_path = self._optimizer_ckp_path(ckp_path)
        torch.save(optimizer.state_dict(), optim_path)
        if scheduler:
            sched_path = self._scheduler_ckp_path(ckp_path)
            torch.save(scheduler.state_dict(), optim_path)
        self.logtrain_string(f'Saved new checkpoint at {ckp_path}')

    def evaluate_fn(self, model, dataloader, loss_fn):
        '''
        >>> model.eval()
        >>> losses = []
        >>> preds = []
        >>> trues = []
        >>> dl_gen = self.dataloader_generator(dataloader)
        >>> for batch in self.tqdm(dl_gen, leave=False, desc='Eval...', total=len(dataloader)):
        >>>     inputs = batch['inputs']
        >>>     targets = batch['targets']
        >>>     with torch.no_grad():
        >>>         o = model(**inputs)
        >>>     loss = self.loss_from_model(o, targets, loss_fn)
        >>>     y = targets['y']
        >>>     preds += o.argmax(1).detach().cpu().numpy().tolist()
        >>>     trues += y.detach().cpu().numpy().tolist()
        >>>     losses.append(loss.item())
        >>> acc = accuracy_score(trues, preds)
        >>> f1 = f1_score(trues, preds)
        >>> conf = confusion_matrix(trues, preds)
        >>> s = '--- Validation ---'
        >>> s += f'\nF1 = {f1}\t Acc = {acc}'
        >>> s += f'\n{conf}'
        >>> self.train_logger.info(s)
        >>> return np.array(losses).mean(), f1
        '''
        raise NotImplementedError

    def arguments_logging(self, args: TrainArgs, model, dl_train, dl_valid, optimizer, loss_fn=None,
                          scheduler=None):
        s = f'Starting training at Absolute Epoch {self.epoch}\n'
        s += f'Train Args: {args}\n'
        s += f'Model:\n{model}\n'
        s += f'Optimizer: {optimizer}\n'
        s += f'Scheduler: {scheduler}\n'
        s += f'Loss Function: {loss_fn}\n'
        s += f'Training Batch Size: {dl_train.batch_size} Sampler: {dl_train.batch_sampler}\n'
        self.params_logger.info(s)

    def train(self, args: TrainArgs, model, dl_train, dl_valid, optimizer, loss_fn=None,
              scheduler=None):

        device = self.device
        self.arguments_logging(args, model, dl_train, dl_valid, optimizer, loss_fn,
                               scheduler)

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

            # evaluate
            valid_loss, valid_metric = self.evaluate_fn(
                model, dl_valid, loss_fn)

            if self.writer:
                self.writer.add_scalar('loss/valid', valid_loss, self.epoch)
                self.writer.add_scalar(
                    'metric/valid', valid_metric, self.epoch)
                for name, param in model.named_parameters():
                    self.writer.add_histogram(name, param, self.epoch)

            s = f'\nAbsolute Epoch {self.epoch} - Relative Epoch [{ep+1}/{args.num_epochs}]'
            s += f'\nTrain Loss {sum(losses_train)/len(losses_train)}'
            s += f'\nValid Loss {valid_loss} Metric {valid_metric}'
            self.logtrain_string(s)

            if valid_metric > self.best_metric:
                self.best_metric = valid_metric
                self.params_logger.info(
                    f'Saved new checkpoint at Absolute Epoch {self.epoch} - Relative Epoch [{ep+1}/{args.num_epochs}]\n'
                    f'with Valid Loss {valid_loss} and metric {valid_metric}')
                self.save_model(model, optimizer, args.ckp_name,
                                scheduler=scheduler)
            self.epoch += 1
