import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from ops.utils import *
from torch.utils.data import DataLoader
from ops.models import TSN
from ops.dataset import TSNDataSet
from ops.transforms import *


class BiasCorrector(nn.Module):
    def __init__(self, alpha=1.,beta=0., current_task=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.n_new_class = len(current_task)

    def forward(self, x):
        x_o = x[:,:-self.n_new_class]
        x_n = x[:,-self.n_new_class:]
        x_n = self.alpha * x_n + self.beta
        x = torch.cat([x_o, x_n],dim=-1)

        return x

def bias_correction(args, age, total_task, current_head, class_indexer, prefix=None):
    K = args.K
    current_task = total_task[-1]
    exemplar_dict = load_exemplars(args)
    exemplar_list = exemplar_dict[age-1]

    # Construct TSM Models
    model = TSN(args, num_class=current_head, modality='RGB',
            fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
            age=age, cur_task_size=len(current_task))

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
    if args.budget_type == 'fixed':
        exemplar_per_class = args.K//(current_head-len(current_task))
    else:
        exemplar_per_class = args.K
    val_dataset = TSNDataSet(args.root_path, args.train_list[0], current_task, class_indexer, num_segments=args.num_segments,
                new_length=1, modality='RGB',image_tmpl=prefix[0], transform=transform_rgb, dense_sample=args.dense_sample,
                exemplar_list=exemplar_list, is_entire=(args.store_frames=='entire'),
                bic='val', nb_val=args.nb_val,exemplar_per_class=exemplar_per_class, current_head=current_head)

    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=args.workers,
                            pin_memory=True, drop_last=False)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    bic_model = BiasCorrector(alpha=1.,beta=0., current_task=current_task).cuda()

    optimizer = torch.optim.SGD(bic_model.parameters(), lr=args.fine_tune_lr)

    logits = []
    labels = []

    # Extract Logits
    with torch.no_grad():
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            outputs = model(input=input)

            logits.append(outputs['preds'])
            labels.append(target)

    logits = torch.cat(logits).cuda()
    labels = torch.cat(labels).cuda()

    for epoch in range(args.start_epoch, args.fine_tune_epochs):
        corrected_logits = bic_model(logits)
        loss = F.cross_entropy(corrected_logits, labels)
        loss.backward()
        print(bic_model.alpha.data, bic_model.beta.data)
        optimizer.step()

    return bic_model

