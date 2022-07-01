import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
import datetime
import time
import os
import math

from ops.models import TSN
from ops.dataset import TSNDataSet
from ops.transforms import *
from ops.utils import *
import cl_methods.cl_utils as cl_utils
import cl_methods.distillation as cl_dist
import cl_methods.classifer as classifier
import copy
from collections import OrderedDict


def _train(args, train_loader, model, criterion, optimizer, epoch, age, lambda_0=[0.5,0.5], model_old=None):
    """
    Train the model only with RGB
    = Not using Flow
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    loss = torch.tensor(0.).cuda()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        outputs = model(input=input)
        preds = outputs['preds']

        if args.loss_type == 'bce':
            target_ = cl_utils.convert_to_one_hot(target, preds.size(1))
        else:
            target_ = target

        if args.fc == 'lsc':
            loss = cl_dist.nca_loss(preds, target_)
        elif args.loss_type == 'nll':
            loss = criterion(preds, target_)
        elif args.loss_type == 'bce':
            target_ = target_[..., old_size:]
            loss = criterion(preds[...,old_size:],target_)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(preds.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(datetime.datetime.now())
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,
                lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)


def _validate(args, val_loader, model, criterion, age):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            outputs = model(input=input)
            output = outputs['preds']

            target_one_hot = cl_utils.convert_to_one_hot(target, output.size(1))

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    top1=top1, top5=top5))

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} '
              .format(top1=top1, top5=top5))
    print(output)

    return top1.avg

def train_task(args, age, total_task, current_head, class_indexer, prefix=None):

    K = args.K
    current_task = total_task[-1]
    exemplar_dict = load_exemplars(args)
    exemplar_list = exemplar_dict[age]

    # Construct TSM Models
    model = TSN(args, num_class=current_head,
            fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
            age=age, cur_task_size=len(current_task),training=True,fine_tune=True)

    scale_size = model.scale_size
    input_size = model.input_size

    normalize = GroupNormalize(model.input_mean, model.input_std)
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)
    transform_rgb = torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])

    print("Load the Model")
    ckpt_path = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}'.format(args.exp), 'task_{:03d}.pth.tar'.format(age))
    sd = torch.load(ckpt_path)
    sd = sd['state_dict']
    state_dict = dict()
    for k, v in sd.items():
        state_dict[k[7:]] = v

    model.load_state_dict(state_dict)

    print(model.new_fc)

    # Construct DataLoader
    task_so_far = [c for task in total_task for c in task]
    train_dataset = TSNDataSet(args.root_path, args.train_list, task_so_far, class_indexer, num_segments=args.num_segments,
                new_length=1, modality='RGB',image_tmpl = prefix, transform=transform_rgb, dense_sample=args.dense_sample,
                exemplar_list=exemplar_list, exemplar_only=True, is_entire=(args.store_frames=='entire'))

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=args.workers,
                            pin_memory=True, drop_last=True)

    print("DataLoader CBF Constructed : Train {}".format(len(train_loader)))

    policies = model.get_cbf_optim_policies()

    # Wrap the model with DataParallel module
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.fine_tune_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.loss_type == 'nll':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss().cuda()

    print("Optimizer Constructed")

    best_prec1 = 0
    best_epoch = 0
    # Train the model for the current task
    for epoch in range(args.start_epoch, args.fine_tune_epochs):
        _adjust_learning_rate(args, optimizer, epoch, args.lr_type, args.lr_steps)
        _train(args, train_loader, model, criterion, optimizer, epoch, age)
        if args.fc in ['cc','lsc']:
                print('Sigma : {}, Eta : {}'.format(model.module.new_fc.sigma, model.module.new_fc.eta))
        state = {'epoch': epoch+1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
        save_checkpoint(args, age, state)

    del model

def _adjust_learning_rate(args, optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


