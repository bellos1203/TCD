import torch
import numpy as np
import os
import shutil
import pickle

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_nme(pred, target, topk=(1,)):
    batch_size = target.size(0)
    pred = pred.t()
    pred = pred.data.cpu()
    correct = pred.eq(target.reshape(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.root_model,
            os.path.join(args.root_model, args.dataset),
            os.path.join(args.root_model, args.dataset, str(args.init_task)),
            os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class)),
            os.path.join(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), '{:03d}'.format(args.exp))
            ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

########################################################

def save_checkpoint(args, age, state, is_best=True):
    filename = '{}/{}/{}/{}/{:03d}/ckpt.pth.tar'.format(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), args.exp)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('ckpt.pth.tar', 'task_{:03d}.pth.tar'.format(age)))

def save_exemplars(args, exemplar_dict): #,exemplar_nframe):
    filename = '{}/{}/{}/{}/{:03d}/exemplars.pth'.format(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), args.exp)
    torch.save(exemplar_dict, filename)

def load_exemplars(args):
    filename = '{}/{}/{}/{}/{:03d}/exemplars.pth'.format(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), args.exp)
    exemplar_dict = torch.load(filename)

    return exemplar_dict

def save_importance(args, age, model, importance_dict):
    importance_list = [model.module.base_model.layer1_importance.importance,
                       model.module.base_model.layer2_importance.importance,
                       model.module.base_model.layer3_importance.importance,
                       model.module.base_model.layer4_importance.importance,
                       model.module.base_model.raw_features_importance.importance]

    importance_dict[age] = importance_list
    filename = '{}/{}/{}/{}/{:03d}/importance.pth'.format(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), args.exp)
    torch.save(importance_dict, filename)

def load_importance(args):
    filename = '{}/{}/{}/{}/{:03d}/importance.pth'.format(args.root_model, args.dataset, str(args.init_task), str(args.nb_class), args.exp)
    importance_dict = torch.load(filename)

    return importance_dict


