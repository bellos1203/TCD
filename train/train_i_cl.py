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
from cl_methods import tcd
import copy

def _train(args, train_loader, model, criterion, optimizer, epoch, age, lambda_0=[0.5,0.5], model_old=None, importance_list=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses_ce = AverageMeter()
    losses_kd_logit = AverageMeter()
    losses_att = AverageMeter()
    losses_div = AverageMeter()

    top1 = AverageMeter()

    loss = torch.tensor(0.).cuda()
    loss_kd_logit = torch.tensor(0.).cuda()
    loss_att = torch.tensor(0.).cuda()
    loss_div = torch.tensor(0.).cuda()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    if model_old:
        model_old.eval()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        outputs = model(input=input,t_div=args.t_div)
        preds = outputs['preds']
        feat = outputs['feat']
        int_features = outputs['int_features']

        if args.loss_type == 'bce':
            target_ = cl_utils.convert_to_one_hot(target, preds.size(1))
        else:
            target_ = target

        if age > 0:
            if args.cl_type == 'DIST':
                with torch.no_grad():
                    outputs_old = model_old(input=input)
                    preds_old = outputs_old['preds']
                    feat_old = outputs_old['feat']
                    int_features_old = outputs_old['int_features']

                old_size = preds_old.size(1)
                preds_base = torch.sigmoid(preds_old)

                if args.fc == 'lsc':
                    loss_ce = cl_dist.nca_loss(preds, target_)
                else:
                    loss_ce = criterion(preds, target_)

                loss_kd_logit = cl_dist.lf_dist_tcd(feat,feat_old,factor=importance_list[-1] if importance_list else None)
                loss_att = cl_dist.feat_dist(int_features,int_features_old,args,factor=importance_list[:-1] if importance_list else None)
                loss = lambda_0[0] * loss_ce + lambda_0[1] * loss_kd_logit + args.lambda_1 * loss_att

                #del preds_old, feat_old, int_features_old
                #del feat, int_features
            else:
                loss_ce = criterion(preds, target_)
                loss = loss_ce
        else:
            if args.fc == 'lsc':
                loss_ce = cl_dist.nca_loss(preds, target_)
            else:
                loss_ce = criterion(preds, target_)
            loss = loss_ce

        if args.t_div:
            loss_div = outputs['t_div'].sum()/input.size(0)
            loss = loss + args.lambda_2 * loss_div

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(preds.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        losses_ce.update(loss_ce.item(), input.size(0))
        losses_kd_logit.update(loss_kd_logit.item(),input.size(0))
        losses_att.update(loss_att.item(), input.size(0))
        losses_div.update(loss_div.item(),input.size(0))
        top1.update(prec1[0].item(), input.size(0))

        #del preds

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #printGPUInfo()

        if i % args.print_freq == 0:
            print(datetime.datetime.now())
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss CE {loss_ce.val:.4f} ({loss_ce.avg:.4f})\t'
                      'Loss KD (Logit) {loss_kd_logit.val:.4f} ({loss_kd_logit.avg:.4f})\t'
                      'Loss KD (Feature) {loss_att.val:.4f} ({loss_att.avg:.4f})\t'
                      'Loss DIV {loss_div.val:.4f} ({loss_div.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_ce=losses_ce, loss_kd_logit=losses_kd_logit,
                loss_att=losses_att, loss_div=losses_div,
                top1=top1, lr=optimizer.param_groups[-1]['lr'] * 0.1))
            print(output)
        #torch.cuda.empty_cache()


def _validate(args, val_loader, model, criterion, age):
    batch_time = AverageMeter()
    top1 = AverageMeter()

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
            prec1 = accuracy(output.data, target, topk=(1,))

            top1.update(prec1[0].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    top1=top1))

    output = ('Testing Results: Prec@1 {top1.avg:.3f} '
              .format(top1=top1))
    print(output)

    return top1.avg

def train_task(args, age, current_task, current_head, class_indexer, model_old=None, prefix=None):
    if age == 1 and args.dataset=='hmdb51' and args.fc=='cc':
        args.lr = args.lr * 0.1
    hook = False
    if age > 0 and args.cl_type=='DIST':
        hook = True
    if age > 0 and args.exemplar:
        exemplar_dict = load_exemplars(args)
        exemplar_list = exemplar_dict[age-1]
        exemplar_per_class = len(exemplar_list[0])

    else:
        exemplar_dict = {}
        exemplar_list = None
        exemplar_per_class = 0

    if age > 0 and args.use_importance:
        importance_dict = load_importance(args)
        importance_temp = importance_dict[age-1]
        importance_list = []
        for i in importance_temp:
            importance_list.append(i)
    else:
        importance_dict = {}
        importance_list = None

    # Construct TSM Models
    model = TSN(args, num_class=current_head if age==0 else current_head-len(current_task), #modality='RGB',
            fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
            apply_hooks=hook, age=age, training=True,
            cur_task_size=len(current_task))#,exemplar_segments=args.exemplar_segments)

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
    lambda_0 = 0.0

    if age > 0:
        print("Load the Previous Model")
        ckpt_path = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}'.format(args.exp), 'task_{:03d}.pth.tar'.format(age-1))

        print(ckpt_path)
        sd = torch.load(ckpt_path)
        sd = sd['state_dict']
        state_dict = dict()
        for k, v in sd.items():
            state_dict[k[7:]] = v

        model.load_state_dict(state_dict)

        # Prepare Old Model to Distill
        if args.cl_type == 'DIST' or args.cl_type == 'REG':
            print("Copy the old Model")
            model_old = copy.deepcopy(model)
            model_old.eval()
            lambda_0 = cl_utils.set_lambda_0(len(current_task),current_head-len(current_task),args)
            print('lambda_0  : {}'.format(lambda_0))

        # Increment the classifier 
        print("Increment the Model")
        model.increment_head(current_head,age)
    print(model.new_fc)
    #print(args.train_list)
    #print(prefix)
    # Construct DataLoader
    train_dataset = TSNDataSet(args.root_path, args.train_list, current_task, class_indexer, num_segments=args.num_segments,
                new_length=1, modality='RGB',image_tmpl=prefix, transform=transform_rgb, dense_sample=args.dense_sample,
                exemplar_list=exemplar_list, is_entire=(args.store_frames=='entire'), #nb_val=args.nb_val,
                exemplar_per_class=exemplar_per_class, current_head=current_head,
                diverse_rate=args.diverse_rate, cl_method=args.cl_method, age=age)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=args.workers,
                            pin_memory=True, drop_last=True)

    print("DataLoader Constructed : Train {}".format(len(train_loader)))

    policies = model.get_optim_policies()

    # Wrap the model with DataParallel module
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if model_old:
        model_old = torch.nn.DataParallel(model_old, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.loss_type == 'nll':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss().cuda()

    print("Optimizer Constructed")

    if age > 0 and args.fc in ['cc','lsc']:
        transform_ex = torchvision.transforms.Compose([
                                                GroupScale(scale_size),
                                                GroupCenterCrop(input_size),
                                                Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                                normalize,
                                                ])

        dataset_for_embedding = TSNDataSet(args.root_path, args.train_list, current_task, class_indexer,
                                num_segments=args.num_segments, random_shift=False, new_length=1,
                                modality='RGB', image_tmpl=prefix, transform=transform_ex,
                                dense_sample=args.dense_sample)


        loader_for_embedding = DataLoader(dataset_for_embedding, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True, drop_last=False)


        cl_utils.init_cosine_classifier(model,current_task,class_indexer,loader_for_embedding,args)

    best_prec1 = 0
    best_epoch = 0
    # Train the model for the current task
    for epoch in range(args.start_epoch, args.epochs):
        _adjust_learning_rate(args, optimizer, epoch, args.lr_type, args.lr_steps)
        print("Learning rate adjusted")

        _train(args, train_loader, model, criterion, optimizer, epoch, age,
                lambda_0=lambda_0, model_old=model_old, importance_list=importance_list)
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            if args.fc in ['cc','lsc']:
                print('Sigma : {}, Eta : {}'.format(model.module.new_fc.sigma, model.module.new_fc.eta))

            state = {'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,}
            save_checkpoint(args, age, state)

    if args.use_importance:
        tcd.update_importance(args,model,train_loader,criterion)
        save_importance(args,age,model,importance_dict)

    torch.cuda.empty_cache()
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


