import torch
import torch.nn as nn
import torch.nn.functional as F
import cl_methods.cl_utils as cl_utils
from math import sqrt


def nca_loss(similarities, targets, exclude_pos_denominator=True,
            hinge_proxynca=False, class_weights=None):
    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")


def lf_dist_tcd(feat, feat_old, factor=None):
    B,T,C = feat.shape
    feat = F.normalize(feat, dim=-1,p=2)
    feat_old = F.normalize(feat_old, dim=-1,p=2)
    feat = feat.view(-1,T*C)
    feat_old = feat_old.view(-1,T*C)

    if factor is not None:
        factor = factor.reshape([1,-1])
        loss_dist = torch.mean(torch.sum(factor*((feat-feat_old)**2),1))
    else:
        loss_dist = torch.mean(torch.sum(((feat-feat_old)**2),1))
    loss_dist = loss_dist/T
    return loss_dist


def feat_dist(fmap, fmap_old, args, factor=None):
    num_layers = len(fmap)
    loss_dist = torch.tensor(0.).cuda(non_blocking=True)
    for i in range(num_layers):
        f1 = fmap[i]
        f1 = f1.view(-1,args.num_segments,f1.size()[1],f1.size()[2],f1.size()[3]) # (B,T,C,H,W)
        f1 = f1.permute(0,2,1,3,4) # (B,C,T,H,W)
        f2 = fmap_old[i]
        f2 = f2.view(-1,args.num_segments,f2.size()[1],f2.size()[2],f2.size()[3])
        f2 = f2.permute(0,2,1,3,4)
        f1 = f1.pow(2)
        f2 = f2.pow(2)
        assert (f1.shape == f2.shape)

        B,C,T,H,W = f1.shape
        f1 = f1.reshape((B,C*T,-1))
        f2 = f2.reshape((B,C*T,-1))
        f1 = F.normalize(f1, dim=2, p=2)
        f2 = F.normalize(f2, dim=2, p=2)
        if factor is not None: # Ours
            if len(factor[i].size()) > 1:
                factor_i = factor[i].permute(1,0)
            else:
                factor_i = factor[i]
            factor_i = factor_i.reshape([1,-1])
            loss_i = torch.mean(factor_i * torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
        else:
            loss_i = torch.mean(torch.frobenius_norm(f1-f2.clone().detach(),dim=-1))
        loss_i = loss_i/sqrt(T)

        loss_dist = loss_dist + loss_i

    loss_dist = loss_dist/num_layers

    return loss_dist

