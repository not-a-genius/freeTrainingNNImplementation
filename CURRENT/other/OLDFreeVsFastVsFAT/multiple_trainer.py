import sys
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, 'lib')
from utils import *
sys.path.insert(1,'train')
from train_free import *
from train_fgsm import *
from train_fat import *

# from train_fgsm import *

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from models import *

from torch.autograd import Variable
import math
import numpy as np
from validation import validate, validate_pgd
from pytorchtools import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
                    
    parser.add_argument('--output_prefix', default='free_adv', type=str,
                    help='prefix used to define output path')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                    help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--earlystop', dest='earlystoppable',default=False, action='store_true',
                    help='use earlystop to stop training based on loss')
    parser.add_argument('--train-on', default=['free', 'fgsm', 'fat'],nargs='+', choices=['free', 'fgsm', 'fat'],
                    help='adv. training algo. to train, then compare: %(default)s)') 
    parser.add_argument('--net', default="resnet50",nargs='?', choices=['smallcnn', 'resnet50', 'WRN','WRN_madry'],
                    help='Pretrained Network to use : %(default)s)') 
    parser.add_argument('--depth', type=int, default=32,
     help='WRN depth')
    parser.add_argument('--width_factor', type=int, default=10, 
    help='WRN width factor')

    parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')

    return parser.parse_args()


configs = parse_config_file(parse_args())


print('==> Load Dataset')
# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(configs.DATA.cifar10_mean, configs.DATA.cifar10_std)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(configs.DATA.cifar10_mean, configs.DATA.cifar10_std)
])
train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=configs.DATA.batch_size, shuffle=True,
    num_workers=configs.DATA.workers, pin_memory=True, sampler=None)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(
    testset, batch_size=configs.DATA.batch_size, shuffle=False,
    num_workers=configs.DATA.workers, pin_memory=True)


net=""
print('==> Load Model')
if configs.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if configs.net == "resnet50":
    model = ResNet50().cuda()
    net = "resnet50"
if configs.net == "WRN":
    model = Wide_ResNet(depth=configs.depth, num_classes=10, widen_factor=configs.width_factor, dropRate=configs.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(configs.depth, configs.width_factor, configs.drop_rate)
if configs.net == 'WRN_madry':
    model = Wide_ResNet_Madry(depth=configs.depth, num_classes=10, widen_factor=configs.width_factor, dropRate=configs.drop_rate).cuda()
    net = "WRN_madry{}-{}-dropout{}".format(configs.depth, configs.width_factor, configs.drop_rate)
print(f"==> Using network: {net}")

model = torch.nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(), configs.TRAIN.lr,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)
criterion = nn.CrossEntropyLoss().cuda()


# Resume if a valid checkpoint path is provided
if configs.resume:
    if os.path.isfile(configs.resume):
        print("=> Loading checkpoint '{}'".format(configs.resume))
        checkpoint = torch.load(configs.resume)
        configs.TRAIN.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Loaded checkpoint '{}' (epoch {})"
                .format(configs.resume, checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(configs.resume))


delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
delta.requires_grad = True

lr_steps = args.epochs * len(train_loader) * args.minibatch_replays
if args.lr_schedule == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
        step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
elif args.lr_schedule == 'multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)



trainer={'free':train_free, 'fgsm':train_fgsm, 'fat':train_fat}
for current_training_algo in configs['train_on']:
    print(f"Trainer, {current_training_algo} \n")
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        
        trainer[current_training_algo]()

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)