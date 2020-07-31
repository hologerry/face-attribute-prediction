'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
import argparse
import os
from os.path import join as ospj
import time

import torch
import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
from celeba import CelebAEval
from utils import AverageMeter, Bar, accuracy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-r', '--root', type=str, default='/D_data/Face_Editing/face_editing/experiments')
parser.add_argument('--exp', type=str, default='face_editing_spectral_attr_n_atten_seg_cyc_g_spade_d_ddp')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='checkpoints/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Attributes
parser.add_argument('--selected_attrs', type=list,
                    default=['Arched_Eyebrows', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses',
                             'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                             'No_Beard', 'Smiling', 'Young', 'Skin_0', 'Skin_1', 'Skin_2', 'Skin_3'])
parser.add_argument('--all_attrs', type=list,
                    default=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                             'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                             'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                             'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                             'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                             'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                             'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                             'Wearing_Necktie', 'Young', 'Skin_0', 'Skin_1', 'Skin_2', 'Skin_3'])

# Device options
parser.add_argument('--gpu-id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

best_prec1 = 0


def main():
    global args, best_prec1, org_idexs2selected_idxes
    args = parser.parse_args()

    args.distributed = False

    org_idexs2selected_idxes = []
    for attr in args.all_attrs:
        if attr in args.selected_attrs:
            org_idexs2selected_idxes.append(args.selected_attrs.index(attr))
        else:
            org_idexs2selected_idxes.append(-1)

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_attributes=44)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            baseWidth=args.base_width,
            cardinality=args.cardinality,
            num_attributes=44,
        )
    elif args.arch.startswith('shufflenet'):
        model = models.__dict__[args.arch](groups=args.groups, num_attributes=44)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_attributes=44)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        args.checkpoint = os.path.dirname(args.resume)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    eval_loader = torch.utils.data.DataLoader(
        CelebAEval(args.root, args.exp, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    _, pred_avg, attrs_top_avg = validate(eval_loader, model, criterion)
    with open(ospj(args.root, args.exp, 'attribute_prediction.txt'), 'w') as f:
        f.write("All selected attrs average accuracy: " + str(pred_avg)+'\n')
        for i, (attr, avg) in enumerate(zip(args.selected_attrs, attrs_top_avg)):
            f.write(f"Attribute {attr} accuracy: {avg}\n")
    print(f"Experiment {args.exp} attribute classifier average accuracy:", pred_avg)


def validate(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(len(args.selected_attrs))]
    top1 = [AverageMeter() for _ in range(len(args.selected_attrs))]

    # switch to evaluate mode
    model.eval()
    loss_avg = 0
    prec1_avg = 0
    top1_avg = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            loss = []
            prec1 = []
            for j in range(len(output)):
                idx = org_idexs2selected_idxes[j]
                if idx != -1:
                    loss.append(criterion(output[j], target[:, idx]))
                    prec1.append(accuracy(output[j], target[:, idx], topk=(1,)))
                    losses[idx].update(loss[-1].item(), input.size(0))
                    top1[idx].update(prec1[-1][0].item(), input.size(0))

            losses_avg = [losses[k].avg for k in range(len(losses))]
            top1_avg = [top1[k].avg for k in range(len(top1))]
            loss_avg = sum(losses_avg) / len(losses_avg)
            prec1_avg = sum(top1_avg) / len(top1_avg)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                batch=i+1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=loss_avg,
                top1=prec1_avg,
            )
            bar.next()
        bar.finish()
    return (loss_avg, prec1_avg, top1_avg)


if __name__ == '__main__':
    main()
