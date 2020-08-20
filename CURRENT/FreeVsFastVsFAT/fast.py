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
from preact_resnet import PreActResNet18

LOAD_WEIGHTS=True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                    help='path to the config file (default: configs.yml)')
    parser.add_argument('--output_prefix', default='fast_adv', type=str,
                    help='prefix used to define output path')
    return parser.parse_args()

args = get_args()
configs = parse_config_file(args)
logger = initiate_logger(configs.output_name, "fast")
print = logger.info

# Check if there is cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    print(configs.TRAIN.epochs)

    # Create output folder
    if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
        os.makedirs(os.path.join('trained_models', configs.output_name))

    # Create weights folder
    if not os.path.isdir(os.path.join('train_fgsm_output', configs.output_name)):
        os.makedirs(os.path.join('train_fgsm_output', configs.output_name))

    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in configs.items(): print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, configs.DATA.batch_size, configs.DATA.workers, configs.DATA.crop_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    print("=> creating model '{}'".format(configs.TRAIN.arch))
    # model = models.__dict__[configs.TRAIN.arch]()
    
    
    model = PreActResNet18().cuda()
    if(LOAD_WEIGHTS):
        logger.info(pad_str("LOADING WEIGHTS"))
        model_path = "cifar_model_weights_30_epochs.pth"
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model = model.eval()
        
    
    
    # Use GPU or CPU
    model = model.to(device)

    
    if configs.evaluate:
        logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            logger.info(pad_str("PGD-" + str(pgd_param[0])))
            pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, pgd_param[0], 10)
            test_loss, test_acc = evaluate_standard(test_loader, model)

            logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
            logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
            return

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

    for pgd_param in configs.ADV.pgd_attack:
        logger.info(pad_str("PGD-" + str(pgd_param[0])))
        pgd_loss, pgd_acc = evaluate_pgd(testloader, model_test, pgd_param[0], 10)
        test_loss, test_acc = evaluate_standard(testloader, model_test)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)

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
    train_err = 0
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
        train_loss += loss.item() * X.shape[0]
        train_acc += (output.max(1)[1] == y).sum().item()
        train_err += (output.max(1)[1] != y).sum().item()
        train_n += y.size(0)
        scheduler.step()
    print("Accuracy: %.3f, Error: %.3f, Loss: %.3f" %(train_acc / len(train_loader.dataset), train_err / len(train_loader.dataset), train_loss / len(train_loader.dataset)))


if __name__ == "__main__":
    main()
