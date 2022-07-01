import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import numpy as np
import cv2
import os
from math import ceil, floor, sqrt, factorial
from cl_methods.classifer import compute_class_mean
from sklearn.cluster import KMeans
import csv
import pandas as pd

def set_task(args, task_list, num_class):
    init_task = args.init_task
    num_task = args.num_task
    classes_per_task = args.nb_class
    dataset = args.dataset

    total_task_list = []

    current_task = task_list[:init_task]
    total_task_list.append(current_task)
    task_so_far = init_task

    for age in range(1, num_task):
        current_task = task_list[task_so_far:task_so_far+classes_per_task]
        total_task_list.append(current_task)
        task_so_far += classes_per_task

    return total_task_list

def set_task_visual(cfg,age,task_list,class_per_task,mode):
    dataset = cfg.TEST.DATASET

    if mode == 'gt':
        cfg.TASK.CURRENT_HEAD = (age+1)*class_per_task
    elif mode =='weight':
        cfg.TASK.CURRENT_HEAD = age*class_per_task

    if dataset == 'kinetics':
        cfg.TASK.CURRENT_TASK = task_list[age*class_per_task:(age+1)*class_per_task]
        cfg.TASK.TASK_SO_FAR = task_list[:(age+1)*class_per_task]

    elif dataset == 'ucf101' or dataset == 'hmdb51':
        cfg.TASK.CURRENT_HEAD = cfg.TASK.CURRENT_HEAD + 1
        cfg.TASK.TASK_SO_FAR = task_list[:(age+1)*class_per_task+1]
        if age == 0:
            cfg.TASK.CURRENT_TASK = task_list[:class_per_task+1]
        else:
            cfg.TASK.CURRENT_TASK = task_list[age*class_per_task+1:(age+1)*class_per_task+1]


def modify_fc_dict(sd, in_features, out_features):
    old_size = sd.size(0)
    if len(sd.size()) > 1: # Weight 
        new_head = torch.normal(mean=0, std=1e-3, size=[out_features, in_features])
    else: # Bias
        new_head = torch.zeros(out_features)
    new_head[:old_size] = sd
    return new_head


def convert_to_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0],num_class).cuda()
    one_hot = one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot


def set_lambda_0(n_new,n_old,args):
    if args.lambda_0_type == 'fix':
        lambda_0 = [1.0, args.lambda_0]
    elif args.lambda_0_type == 'arith':
        lambda_0 = [n_new/(n_old+n_new), n_old/(n_old+n_new)]
    elif args.lambda_0_type == 'geo':
        lambda_0 = [1.0, args.lambda_0*sqrt(n_old/n_new)]
    return lambda_0


def init_cosine_classifier(model,current_task,class_indexer,dloader,args):
    print('Initialize Cosine Classifier')

    old_embedding_norm = model.module.new_fc.fc1.weight.data.norm(dim=1, keepdim=True)
    average_old_embedding_norm = torch.mean(old_embedding_norm,dim=0).type(torch.FloatTensor)
    num_features = model.module.new_fc.in_features

    cls_embedding, features = compute_class_mean(model,current_task,dloader,class_indexer)
    novel_cls_embedding = cls_embedding * average_old_embedding_norm

    if args.fc == 'cc':
        model.module.new_fc.fc2.weight.data = novel_cls_embedding.data.cuda()
    elif args.fc == 'lsc':
        new_weights = []
        assert cls_embedding.size()[0] == len(features)
        for i in range(len(features)):
            feature_i = F.normalize(features[i],p=2,dim=1)
            clusterizer = KMeans(n_clusters=args.num_proxy)
            clusterizer.fit(feature_i.cpu().numpy())
            for center in clusterizer.cluster_centers_:
                new_weights.append(torch.tensor(center) * average_old_embedding_norm)
        new_weights = torch.stack(new_weights).cuda()

        model.module.new_fc.fc2.weight.data = new_weights

        del new_weights
    del novel_cls_embedding


def stable_cosine_distance(a, b, squared=True):
    # From PODNet
    """Computes the pairwise distance matrix with numerical stability."""
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]

def factorize(feat,dim):
    """
    dim = 2(T),3(H),4(W)
    """
    B,C,T,H,W = feat.shape

    feat = feat.sum(dim=dim) # B,C,d1,d2
    if not isinstance(dim,tuple):
        feat = feat/(feat.norm(dim=(2,3),keepdim=True) + 1e-8)
    else:
        feat = feat/(feat.norm(dim=-1,keepdim=True) + 1e-8)
    feat = feat.view(B,-1)
    return feat

def _compute_avg_total(arr,n_vids):
    #avg = np.average(arr,weights=(arr > -0.5), axis=0)
    n_vids = np.array(n_vids)
    n_vids_pad = np.concatenate((n_vids,np.zeros(np.shape(arr)[0]-len(n_vids)))).reshape(-1,1)
    tmp = arr*n_vids_pad
    tmp = np.triu(tmp)
    avg = np.sum(tmp,axis=0)/np.cumsum(n_vids)

    return avg

def _compute_avg(arr):
    avg = np.average(arr,weights=(arr > -0.5), axis=0)
    return avg

def _compute_fgt(arr,n_vids):
    n_vids = np.array(n_vids)
    n_vids_pad = np.concatenate((n_vids,np.zeros(np.shape(arr)[0]-len(n_vids)))).reshape(-1,1)
    tmp = arr*n_vids_pad
    tmp = -np.diff(tmp)[:-1]
    tmp = np.triu(tmp)
    tmp = np.sum(tmp,axis=0)
    cumsum_n_vids = np.cumsum(n_vids)[:-1]
    fgt = tmp/cumsum_n_vids
    fgt = np.mean(fgt)
    fgt = fgt[np.newaxis]

    return fgt

def compute_final_stats(n_videos, args, cls='cnn'):
    csv_final = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}.csv'.format(args.exp))

    csv_results = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}'.format(args.exp), '{:03d}_{}.csv'.format(args.exp,cls))
    results = pd.read_csv(csv_results, header=None)
    results = np.array(results)
    results = np.transpose(np.array(results))
    avg_top1_tot = _compute_avg_total(results,n_videos)
    #avg_top5_tot = _compute_avg_total(result_top5,n_videos)
    # forgetting
    fgt_top1 = _compute_fgt(results,n_videos)
    #fgt_top5 = _compute_fgt(result_top5,n_videos)
    top1 = np.concatenate((fgt_top1,avg_top1_tot))
    #top5 = np.concatenate((fgt_top5,avg_top5,avg_top5_tot))

    csv_file = os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}.csv'.format(args.exp))
    with open(csv_file, mode='a') as r_file:
        r_writer = csv.writer(r_file, delimiter=',')
        r_writer.writerow(top1)
        #r_writer.writerow(top5)

