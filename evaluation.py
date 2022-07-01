import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import datetime
import time
import os
import csv

from ops.models import TSN
from ops.dataset import TSNDataSet
from ops.transforms import *
from ops.utils import *
from cl_methods.classifer import nme, compute_class_mean
import copy
import shutil

def _test(args, test_loader, model, bic_model=None):
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    std = []
    end = time.time()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(test_loader):
            num_crop = args.test_crops # 10 * 2 # twice_sample
            if args.dense_sample:
                num_crop *= 10
            if args.twice_sample:
                num_crop *= 2
            length = 3 # RGB
            batch_size = target.numel()
            input_in = input.view(-1, length, input.size(2), input.size(3)).cuda()

            if args.shift:
                input_in = input_in.view(batch_size * num_crop, args.num_segments, length, input_in.size(2), input_in.size(3)).cuda()
            target = target.cuda()

            # compute output
            outputs = model(input_in)
            output = outputs['preds']

            del outputs

            output = output.reshape(batch_size, num_crop, -1).mean(1)
            if bic_model:
                output = bic_model(output)
            output = F.softmax(output, dim=1)

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
                    i, len(test_loader), batch_time=batch_time,
                    top1=top1))
                print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f}'
              .format(top1=top1))
    print(output)

    torch.cuda.empty_cache()
    return top1.avg

def _record_results(args, age, results, cls='cnn'):
    csv_file = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}'.format(args.exp), '{:03d}_{}.csv'.format(args.exp,cls))
    prev_results = None
    if os.path.exists(csv_file) and age > 0:
        with open(csv_file, mode='r') as r_file:
            r_reader = csv.reader(r_file)
            prev_results = [row for idx, row in enumerate(r_reader) if idx < age]

    with open(csv_file, mode='w') as r_file:
        r_writer = csv.writer(r_file, delimiter=',')
        if prev_results:
            r_writer.writerows(prev_results)
        r_writer.writerow(results)


def eval_task(args, age, task_list, current_head, class_indexer, cur_task_size, prefix=None, bic_model=None):
    # Construct TSM Models
    model = TSN(args, num_class=current_head,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                age=age,cur_task_size=cur_task_size)

    if args.exemplar:
        exemplar_dict  = load_exemplars(args)
        exemplar_list = exemplar_dict[age]
    else:
        exemplar_list = None

    input_size = model.input_size
    crop_size = model.crop_size
    scale_size = model.scale_size
    normalize = GroupNormalize(model.input_mean, model.input_std)
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
                GroupOverSample(input_size, scale_size)
        ])

    transform = torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    ckpt_path = os.path.join(args.root_model, args.dataset, str(args.init_task),  str(args.nb_class), '{:03d}'.format(args.exp), 'task_{:03d}.pth.tar'.format(age))
    print("Load the Trained Model from {}".format(ckpt_path))
    sd = torch.load(ckpt_path)
    sd = sd['state_dict']
    model.load_state_dict(sd)

    model.eval()

    if args.exemplar and args.cl_method != 'GDumb':
        task_so_far = [c for task in task_list for c in task]
        transform_ex = torchvision.transforms.Compose([
                                                GroupScale(scale_size),
                                                GroupCenterCrop(input_size),
                                                Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                                normalize,
                                                ])

        ex_dataset = TSNDataSet(args.root_path, args.train_list, task_so_far, class_indexer,
                                num_segments=args.num_segments, random_shift=False, new_length=1,
                                modality='RGB',image_tmpl = prefix, transform=transform_ex,
                                dense_sample=args.dense_sample, exemplar_list=exemplar_list,
                                exemplar_only=True, is_entire=(args.store_frames=='entire'))

        ex_loader = DataLoader(ex_dataset, batch_size=args.exemplar_batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=True, drop_last=False)

        class_means, _ = compute_class_mean(model,current_head,ex_loader)
        nme_results_top1 = []

    results_top1 = []

    num_test_videos = []
    # Construct DataLoader
    for i in range(age+1):
        print("Eval Task {} for Age {}".format(i, age))
        print("Current Task : {}".format(task_list[i]))
        test_dataset = TSNDataSet(args.root_path, args.val_list, task_list[i], class_indexer, num_segments=args.num_segments, random_shift=False,
                    new_length=1, modality='RGB', image_tmpl = prefix, transform=transform, dense_sample=args.dense_sample,
                    test_mode=True, twice_sample=args.twice_sample)
        num_test_videos.append(len(test_dataset.video_list))

        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=True, drop_last=False)
        print("DataLoader Constructed")

        prec1 = _test(args, test_loader, model, bic_model)

        results_top1.append(prec1)

        if args.nme:
            prec1_nme = nme(model,class_means,test_loader,args)
            nme_results_top1.append(prec1_nme)

    if len(results_top1) < args.num_task:
        for i in range(args.num_task-len(results_top1)):
            results_top1.append(-200)

    _record_results(args, age, results_top1, 'cnn')

    if args.nme:
        if len(nme_results_top1) < args.num_task:
            for i in range(args.num_task-len(nme_results_top1)):
                nme_results_top1.append(-200)

        _record_results(args, age, nme_results_top1, 'nme')

    if args.exemplar:
        del class_means, _
    del model

    return num_test_videos

