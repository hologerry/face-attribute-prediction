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
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
from wild import Wild
from utils import AverageMeter, Bar

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
parser.add_argument('-r', '--root', type=str, default='/D_data/Face_Editing/face_editing/data')
parser.add_argument('--exp', type=str, default='wild_images')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize (default: 200)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints_celebahq', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='checkpoints_celebahq/model_best.pth.tar', type=str, metavar='PATH',
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
parser.add_argument('--pred_attr_file', type=str, default='celebahq_pred_attributes_list.txt')
parser.add_argument('--pred_attr_dir', type=str, default='celebahq_attr_prob')

# Device options
parser.add_argument('--gpu-id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

best_prec1 = 0
args = None
org_idexs2selected_idxes = None


def main():
    global args, best_prec1, org_idexs2selected_idxes
    args = parser.parse_args()

    args.distributed = False

    # org_idexs2selected_idxes = []
    # for attr in args.all_attrs:
    #     if attr in args.selected_attrs:
    #         org_idexs2selected_idxes.append(args.selected_attrs.index(attr))
    #     else:
    #         org_idexs2selected_idxes.append(-1)

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
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    eval_loader = torch.utils.data.DataLoader(
        Wild(args.root, args.exp, 'test', transform_img=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    attr_f = open(args.pred_attr_file, 'w')
    attr_f.write(' '.join(args.all_attrs)+'\n')
    attr_prob_dir = ospj(args.root, args.exp, args.pred_attr_dir)

    _, pred_avg, attrs_top_avg = validate(eval_loader, model, criterion, attr_f, attr_prob_dir)
    with open(ospj(args.root, args.exp, 'attribute_prediction.txt'), 'w') as f:
        f.write("All selected attrs average accuracy: " + str(pred_avg)+'\n')
        for i, (attr, avg) in enumerate(zip(args.selected_attrs, attrs_top_avg)):
            f.write(f"Attribute {attr} accuracy: {avg}\n")
    print(f"Experiment {args.exp} attribute classifier average accuracy:", pred_avg)


def validate(val_loader, model, criterion, attribute_f, attribute_prob_dir):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = [AverageMeter() for _ in range(len(args.selected_attrs))]
    # top1 = [AverageMeter() for _ in range(len(args.selected_attrs))]
    # switch to evaluate mode
    model.eval()
    loss_avg = 0
    prec1_avg = 0
    top1_avg = []

    with torch.no_grad():
        end = time.time()
        for i, (input, img_idx) in enumerate(val_loader):
            # measure data loading time
            assert input.size(0) == 1
            data_time.update(time.time() - end)

            img_idx = img_idx.cuda(non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            # print(output[0])
            # print(output[0].size())
            sub_dir = ospj(attribute_prob_dir, f'{(img_idx.item() // 1000):02d}000')
            os.makedirs(sub_dir, exist_ok=True)
            attr_prob_file_name = f'{img_idx.item():05d}.txt'
            attr_path = ospj(sub_dir, attr_prob_file_name)
            attr_f = open(attr_path, 'w')
            img_path = f'{img_idx.item():05d}.png '
            attribute_f.write(img_path)
            for j in range(len(output)):
                _, pred = output[j].topk(1, 1, True, True)
                attr = '1' if pred else '-1'
                attribute_f.write(attr + ' ')
                out = F.softmax(output[j], dim=1)
                val0_prob = out[0, 0].item()
                val1_prob = out[0, 1].item()
                attr_f.write(args.all_attrs[j] + ' ')
                attr_f.write(f'{val0_prob:.6f} ')
                attr_f.write(f'{val1_prob:.6f}\n')
            attribute_f.write('\n')
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
