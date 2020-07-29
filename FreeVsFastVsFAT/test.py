import init_paths
import argparse
import copy
import logging
import math
import random
import sys
import time
from utils import *
import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets

PRECS=[]
class Parameters:

    data_dir ='./data'
    lr_schedule ='cyclic'
    lr_min = 0
    lr_max = 0.2
    weight_decay = 5e-4
    momentum = 0.9
    epsilon = 8    
    alpha = 10
    delta_init = 'random'
    out_dir = 'train_fgsm_output'
    seed = 0
    early_stop = False
    opt_level = 'O2'
    loss_scale = '1.0'
    master_weights = False
    output_prefix = 'fast_adv'
    config = 'configs.yml'
args=Parameters()
configs = parse_config_file(args)
logger = initiate_logger(configs.output_name, "fast")
print = logger.info




def train(train_loader, model, criterion, epoch, epsilon, opt, alpha, scheduler):
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    start_epoch_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    for i, (X, y) in enumerate(train_loader):
        end = time.time()
        X = X.to(device)
        y = y.to(device)
        data_time.update(time.time() - end)
        if i == 0:
            first_batch = (X, y)
        if args.delta_init != 'previous':
            delta = torch.zeros_like(X).cuda()
        if args.delta_init == 'random':
            for j in range(len(epsilon)):
                delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        output = model(X + delta[:X.size(0)])
        loss = F.cross_entropy(output, y)
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
        delta = delta.detach()
        output = model(X + delta[:X.size(0)])
        loss = criterion(output, y)

        opt.zero_grad()
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        opt.step()

        prec1, prec5 = accuracy(output, y, topk=(1, 5))
        print(prec1,prec5)
        PRECS.append((prec1,prec5))
        losses.update(loss.item(), X.size(0))
        top1.update(prec1[0], X.size(0))
        top5.update(prec5[0], X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % configs.TRAIN.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
            sys.stdout.flush()
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        scheduler.step()
        



# Check if there is cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'


print(configs.TRAIN.epochs)

# Create output folder
if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
    os.makedirs(os.path.join('trained_models', configs.output_name))

# Log the config details
logger.info(pad_str(' ARGUMENTS '))
for k, v in configs.items(): print('{}: {}'.format(k, v))
logger.info(pad_str(''))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

train_loader, test_loader = get_loaders(args.data_dir, configs.DATA.batch_size, configs.DATA.workers)

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std
pgd_alpha = (2 / 255.) / std

print("=> creating model '{}'".format(configs.TRAIN.arch))
model = models.__dict__[configs.TRAIN.arch]()

# Use GPU or CPU
model = model.to(device)

model.train()

opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
if args.opt_level == 'O2':
    amp_args['master_weights'] = args.master_weights
model, opt = amp.initialize(model, opt, **amp_args)
criterion = nn.CrossEntropyLoss()

if args.delta_init == 'previous':
    delta = torch.zeros(configs.DATA.batch_size, 3, 32, 32).cuda()

lr_steps = configs.TRAIN.epochs * len(train_loader)
if args.lr_schedule == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
        step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
elif args.lr_schedule == 'multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

# Training
prev_robust_acc = 0.
start_train_time = time.time()

for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
    
    # Train
    train(train_loader, model, criterion, epoch, epsilon, opt, alpha, scheduler)

    if args.early_stop:
        # Check current PGD robustness of model using random minibatch
        X, y = first_batch
        pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
        with torch.no_grad():
            output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
        robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
        if robust_acc - prev_robust_acc < -0.2:
            break
        prev_robust_acc = robust_acc
        best_state_dict = copy.deepcopy(model.state_dict())
    


train_time = time.time()
if not args.early_stop:
    best_state_dict = model.state_dict()
torch.save(best_state_dict, os.path.join(args.out_dir, 'model.pth'))
logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

# Evaluation

model_test = models.__dict__[configs.TRAIN.arch]().to(device)
model_test.load_state_dict(best_state_dict)
model_test.float()
model_test.eval()

pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
test_loss, test_acc = evaluate_standard(test_loader, model_test)

logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
# x = range(100)
# for i in x:
#     writer.add_scalar('y = 2x', i * 2, i)
# writer.close()