import torch
import cl_methods.distillation as cl_dist


def update_importance(args, training_network, train_loader, criterion):
    print('Update Importance Mask...')
    training_network.module.base_model.reset_importance()
    training_network.module.base_model.start_cal_importance()
    for i, (input, target, _) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        outputs = training_network(input)
        preds = outputs['preds']
        if args.fc =='lsc':
            loss = cl_dist.nca_loss(preds, target)
        else:
            # Classification loss is cosine + learned factor + softmax:
            loss = criterion(preds, target)

        loss.backward()

    training_network.module.base_model.stop_cal_importance()
    training_network.module.base_model.normalize_importance()

