import torch

from mlpack.bert.ner.utils import to_device, save_model

try:
    from apex import amp
except ImportError:
    print(
        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def train(args, model, dl_train, dl_valid, optimizer, scheduler=None, evaluate_fn=None, notebook=True):

    if notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    device = args.device

    for ep in tqdm(range(args.num_epochs), desc='Epochs'):
        model.train()
        for step, (input_ids, input_mask, label_ids, label_mask) in tqdm(enumerate(dl_train), leave=False, total=len(dl_train)):
            input_ids, input_mask, label_ids, label_mask = to_device(input_ids, input_mask, label_ids,
                                                                     label_mask, device=device)

            loss, _, _ = model(input_ids, input_mask,
                               label_ids, label_mask)

            if args.grad_steps > 1:
                loss = loss / args.grad_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

            if (step + 1) % args.grad_steps == 0 or (step + 1) == len(dl_train):
                optimizer.step()
                model.zero_grad()
                if scheduler:
                    scheduler.step()

            if args.writer:
                args.writer.add_scalar('loss/train', loss, args.n_iter)
            args.n_iter += 1

        # evaluate
        if evaluate_fn:
            valid_loss, valid_acc = evaluate_fn(model, dl_valid)

        if args.writer:
            args.epoch += ep
            args.writer.add_scalar('loss/valid', valid_loss, args.epoch)
            args.writer.add_scalar('acc/valid', valid_acc, args.epoch)

            for name, param in model.named_parameters():
                args.writer.add_histogram(name, param, args.epoch)

        print(f'---Valid\nLoss {valid_loss}\nAcc {valid_acc}', flush=True)

        if args.best_acc is None:
            args.best_acc = valid_acc
            save_model(model, optimizer, args.ckp_path, scheduler=scheduler)
        else:
            if valid_acc > args.best_acc:
                args.best_acc = valid_acc
                save_model(model, optimizer, args.ckp_path,
                           scheduler=scheduler)
