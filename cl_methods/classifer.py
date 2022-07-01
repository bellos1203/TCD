import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.utils import *
import numpy as np
import time

### https://github.com/imyzx2017/icarl.pytorch/blob/master/icarl.py

def _get_closest(means, features, test_mode, num_crop, target, topk=5):
    num_class = means.size(0)
    if len(features.shape)==3:
        features = features.permute(1,0,2)
        means = means.permute(1,0,2)
    distances = torch.cdist(features,means,p=2)
    if len(distances.shape)==3:
        distances = distances.sum(0)

    if test_mode:
        distances = distances.reshape(-1,num_crop,num_class).mean(1)

    return -distances

def compute_class_mean(model,current_task,exemplar_loader,class_indexer=None):
    print("Computing the class mean vectors...")
    model.eval()
    ex_dict = {}
    std = []
    norm_std = []
    if class_indexer:
        loop_criteria = current_task
        for i in current_task:
            ex_dict[class_indexer[i]] = {}
    else:
        loop_criteria = range(current_task)
        for i in range(current_task):
            ex_dict[i] = {}
    with torch.no_grad():
        for i, (input, target, props) in enumerate(exemplar_loader):
            input = input.cuda()
            target = target.cuda()

            outputs = model(input=input, only_feat=True)
            base_out = outputs['preds']
            feat = outputs['feat'].data.cpu()
            del outputs
            feat = feat.mean(1)

            for j in range(target.size(0)):
                k = props[0][j]
                v = feat[j]
                if int(target[j]) in ex_dict.keys():
                    ex_dict[int(target[j])].update({k:v})

    class_mean_list = []
    feature_list = []

    for i in loop_criteria:
        if class_indexer:
            temp_dict = ex_dict[class_indexer[i]]
        else:
            temp_dict = ex_dict[i]
        features = []

        for k, v in enumerate(temp_dict.items()):
            f_path = v[0]
            feat = v[1]
            feat = feat/torch.norm(feat,p=2)
            features.append(feat)

        features = torch.stack(features)
        class_mean = torch.mean(features,axis=0)
        class_mean = class_mean / torch.norm(class_mean,p=2)

        feature_list.append(features)
        class_mean_list.append(class_mean)

    class_means = torch.stack(class_mean_list)

    return class_means, feature_list


def nme(model,class_means,test_loader,args,test_mode=True):

    model.eval()
    batch_time = AverageMeter()
    top1 = AverageMeter()


    print("Classify using the NME Classifier...")
    end = time.time()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(test_loader):
            if test_mode:
                num_crop = args.test_crops
                if args.dense_sample:
                    num_crop *= 10
                if args.twice_sample:
                    num_crop *= 2
                length = 3
                batch_size = target.numel()
                input_in = input.view(-1,length,input.size(2),input.size(3)).cuda()
                if args.shift:
                    input_in = input_in.view(batch_size*num_crop, args.num_segments, length, input_in.size(2), input_in.size(3)).cuda()
            else:
                input_in = input.cuda()
            outputs = model(input=input_in)
            feat = outputs['feat'].mean(1)

            del outputs

            feat = feat.data.cpu()
            feat = feat / torch.norm(feat,p=2,dim=-1,keepdim=True)

            preds = _get_closest(class_means,feat,test_mode,num_crop,target)

            prec1 = accuracy(preds.data.cpu(), target.cpu(), topk=(1,))

            top1.update(prec1[0].item(), input.size(0))

            del preds, feat

            if i % args.print_freq == 0:
                output = ('Test (NME): [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time,
                    top1=top1))
                print(output)

    output = ('Testing Results (NME): Prec@1 {top1.avg:.3f}'
              .format(top1=top1))
    print(output)

    torch.cuda.empty_cache()
    return top1.avg
