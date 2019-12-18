import torch
import json
import pickle
import os


def save_pickle(obj, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)


def read_json(filepath):
    with open(filepath, 'rb') as file:
        return json.load(file)


def save_json(obj, filepath):
    with open(filepath, 'w') as file:
        json.dump(obj, file)


def create_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)


def to_fp16(model, optimizer):
    try:
        from apex import amp
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer)
    return model, optimizer


def _optimizer_ckp_path(ckp_path):
    fmt = ckp_path.split('/')[-1].split('.')[-1]
    optim_path = ckp_path.replace(f'.{fmt}', f'_optimizer.{fmt}')
    return optim_path


def _scheduler_ckp_path(ckp_path):
    fmt = ckp_path.split('/')[-1].split('.')[-1]
    sched_path = ckp_path.replace(f'.{fmt}', f'_lrscheduler.{fmt}')
    return sched_path


def save_model(model, optimizer, ckp_path, scheduler=None):
    torch.save(model.state_dict(), ckp_path)
    # saving the optimizer
    optim_path = _optimizer_ckp_path(ckp_path)
    torch.save(optimizer.state_dict(), optim_path)
    if scheduler:
        sched_path = _scheduler_ckp_path(ckp_path)
        torch.save(scheduler.state_dict(), optim_path)
    print('Saved new checkpoint', flush=True)


def to_device(*tensors, device='cpu'):
    return [
        t.to(device) for t in tensors
    ]
