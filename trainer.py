import gc
import time
import torch
import logging
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import models
from optimizers import get_optimizer
from utils import *

def contrastive_trainer(train_dataset, args):
    start = time.time()

    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=0,
                                drop_last=True,
                                shuffle=True)

    if args.train_mode == 'clco':
        model = models.clco_model(args)
    else:
        raise "no such model"
    model.to(args.device)

    optimizer = get_optimizer((model,), 'pretrain', args)

    if args.pretrain_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                    T_max=args.max_epochs-args.warm, eta_min=1e-5, 
                    last_epoch=-1)
    if args.loss in ['mpc']:
        criterion = torch.nn.CrossEntropyLoss().to(args.device)

    if args.active_log:
        logging.info(f"Model checkpoint and metadata has been saved at"
                    f"model: {model.__class__}, optimizer: {optimizer.__class__}")
        logging.info(f"Start Model at {time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}")

    best_train_loss = np.inf

    for epoch in range(args.max_epochs):
        data_iter = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc='Epoch %d' % epoch)
        total_loss, total_acc = 0, 0

        if args.pretrain_lr_scheduler:
            # lr scheduler with warmup
            if args.warm > 0 and epoch + 1 <= args.warm:
                wu_lr = (args.lr - args.base_lr) * \
                    (float(epoch + 1) / args.warm) + args.base_lr
                optimizer.param_groups[0]['lr'] = wu_lr
            else:
                scheduler.step()

        for i, ((batch_view_1, batch_view_2), labels) in data_iter:
            optimizer.zero_grad()

            batch_view_1 = batch_view_1.to(args.device)
            batch_view_2 = batch_view_2.to(args.device)

            if args.loss in ['mpc']:
                logit, target = model(batch_view_1, batch_view_2, labels)
                loss = criterion(logit, target)
            elif args.loss in ['supcon', 'mcc']:
                loss, target = model(batch_view_1, batch_view_2, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if args.loss == 'mpc':
                acc = accuracy(logit, target, topk=(1,))
                total_acc += acc[0].item()
                
            if i % 10 == 0 and args.active_log:
                args.writer.add_scalar('train loss', loss.item(), epoch * len(train_loader) + i)
                if args.loss == 'mpc':
                    args.writer.add_scalar('train acc', acc[0].item(), epoch * len(train_loader) + i)
                args.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch *\
                    len(train_loader) + i)
        
        if args.active_log and args.loss == 'mpc':
            logging.debug(f"Epoch: {epoch}/{args.max_epochs}, \tLoss: {total_loss}, \tAcc: \
                          {total_acc / len(train_loader)}")
            print(f"Epoch: {epoch}/{args.max_epochs}, \tLoss: {total_loss}, \tAcc: \
                          {total_acc / len(train_loader)}")
        
        if args.active_log:
            # save checkpoint
            torch.save({
                'model': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
            }, args.checkpoint_dir)
            if total_loss <= best_train_loss:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                }, args.checkpoint_dir[:-3] + "_best.pt")
                best_train_loss = total_loss
            runtime = divmod(time.time() - start, 60)
            logging.info(f"Model checkpoint and metadata has been saved at {args.writer.log_dir},"
                        f" with runtime {runtime[0]}m {runtime[1]}s.")
        gc.collect()

def finetune(model, train_dataset, val_dataset, args):
    ''' Finetune script '''
    train_loader = DataLoader(train_dataset,
                            batch_size=args.finetune_batch_size,
                            num_workers=0,
                            drop_last=False,
                            shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.finetune_batch_size,
                            num_workers=0,
                            drop_last=False,
                            shuffle=False)

    ''' optimizers '''
    nofcparam, fcparam = [], []
    for name, param in model.named_parameters():
        if 'fc' in name:
            fcparam.append(param)
        else:
            nofcparam.append(param)
    optimizer = torch.optim.SGD(nofcparam, lr=args.ftlr,
                              weight_decay=args.ftwd, momentum=0.9)
    fcoptimizer = torch.optim.SGD(fcparam, lr=args.fclr,
                              weight_decay=args.ftwd, momentum=0.9)

    ''' Schedulers '''
    if args.finetune_lr_scheduler:
        # Cosine LR Decay
        lr_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    args.finetune_epochs, eta_min=1e-5)

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # initilize Variables
    best_valid_loss = np.inf

    ''' Pretrain loop '''
    for epoch in range(args.finetune_epochs):
        model.train()
        
        sample_count = 0
        run_loss = 0
        run_top1 = 0.0

        train_dataloader = tqdm(train_loader,
                        total=len(train_loader),
                        desc='Epoch %d' % epoch)

        ''' epoch loop '''
        for i, (inputs, target) in enumerate(train_dataloader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)

            # Forward pass
            optimizer.zero_grad()
            fcoptimizer.zero_grad()

            output = model(inputs)

            # Take pretrained encoder representations
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            fcoptimizer.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)
            run_loss += loss.item()
            acc = accuracy(output, target, topk=(1,))
            run_top1 += acc[0].item()
        
        epoch_finetune_loss = run_loss / len(train_loader)  # sample_count
        epoch_finetune_acc = run_top1 / len(train_loader)
        
        ''' Update Schedulers '''
        if args.finetune_lr_scheduler:
            # Decay lr with CosineAnnealingLR
            lr_decay.step()
        
        valid_loss, valid_acc = evaluate(
            model, val_loader, epoch, args)

        ''' Printing '''
        if args.active_log:
            logging.info('Epoch {}/{}, [Finetune] loss: {:.4f},\t acc: {:.4f}'.format(
                epoch + 1, args.finetune_epochs, epoch_finetune_loss, epoch_finetune_acc))
            print('Epoch {}/{}, [Finetune] loss: {:.4f},\t acc: {:.4f}'.format(
                epoch + 1, args.finetune_epochs, epoch_finetune_loss, epoch_finetune_acc))
            logging.info('\4t[eval] loss: {:.4f},\t acc: {:.4f}'.format(valid_loss, valid_acc))
            print('\4t[eval] loss: {:.4f},\t acc: {:.4f}'.format(valid_loss, valid_acc))
            args.writer.add_scalar('finetune_epoch_loss_train', epoch_finetune_loss, epoch + 1)
            args.writer.add_scalar('finetune_epoch_acc_train', epoch_finetune_acc, epoch + 1)
            args.writer.add_scalar('finetune_lr_train', optimizer.param_groups[0]['lr'], epoch + 1)
            args.writer.add_scalar('finetune_epoch_loss_eval', valid_loss, epoch + 1)
            args.writer.add_scalar('finetune_epoch_acc_eval', valid_acc, epoch + 1)

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss

            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            if args.active_log:
                torch.save(state, (args.checkpoint_dir[:-3] + "_finetune.pt"))

        epoch_finetune_loss = None  # reset loss
        epoch_finetune_acc = None

    del state


def evaluate(model, val_loader, epoch, args):
    epoch_valid_loss = None  # reset loss
    epoch_valid_acc = None  # reset acc

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # Evaluate both encoder and class head
    model.eval()

    # initilize Variables
    sample_count = 0
    run_loss = 0
    run_top1 = 0.0
    
    eval_dataloader = tqdm(val_loader,
                        total=len(val_loader),
                        desc='Epoch %d' % epoch)

    ''' epoch loop '''
    for i, (inputs, target) in enumerate(eval_dataloader):
        model.zero_grad()

        inputs = inputs.to(args.device)
        target = target.to(args.device)

        # Forward pass
        output = model(inputs)
        loss = criterion(output, target)

        torch.cuda.synchronize()

        sample_count += inputs.size(0)
        run_loss += loss.item()
        acc = accuracy(output, target, topk=(1,))
        run_top1 += acc[0].item()

    epoch_valid_loss = run_loss / len(val_loader)
    epoch_valid_acc = run_top1 / len(val_loader)

    ''' Printing '''
    if args.train_mode == "evaluate" and args.active_log:
        logging.info('[eval] loss: {:.4f},\t acc: {:.4f}'.format(epoch_valid_loss, epoch_valid_acc))
        print('[eval] loss: {:.4f},\t acc: {:.4f}'.format(epoch_valid_loss, epoch_valid_acc))
    return epoch_valid_loss, epoch_valid_acc
