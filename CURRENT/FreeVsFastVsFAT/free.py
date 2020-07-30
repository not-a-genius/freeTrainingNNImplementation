import init_paths
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from utils import *
from validation import validate, validate_pgd
import torchvision.models as models

'''
    Arguments 
'''
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
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
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    return parser.parse_args()

# Parse config file and initiate logging
args = parse_args()
configs = parse_config_file(args)
logger = initiate_logger(configs.output_name, "free")
print = logger.info
cudnn.benchmark = True

# Check if there is cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")
def main():
    # Scale and initialize the parameters
    best_prec1 = 0
    configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs / configs.ADV.n_repeats))
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value

    # Create output folder
    if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
        os.makedirs(os.path.join('trained_models', configs.output_name))

    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in configs.items(): print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    # Create the model
    if configs.pretrained:
        print("=> using pre-trained model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch]()


    # Use GPU or CPU
    model = model.to(device)
    
    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), configs.TRAIN.lr,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    # Data
    print('==> Preparing data..')

    trainloader, testloader = get_loaders("./data", configs.DATA.batch_size, configs.DATA.workers, configs.DATA.crop_size)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(testloader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)
        validate(testloader, model, criterion, configs, logger)
        return
    
    start_train_time = time.time()
    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        adjust_learning_rate(configs.TRAIN.lr, optimizer, epoch, configs.ADV.n_repeats)

        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(testloader, model, criterion, configs, logger)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': configs.TRAIN.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, os.path.join('trained_models', configs.output_name))
    
    train_time = time.time()
    print('Total train time: %.4f minutes', (train_time - start_train_time)/60)
   
    # Automatically perform PGD Attacks at the end of training
    logger.info(pad_str(' Performing PGD Attacks '))
    '''
    for pgd_param in configs.ADV.pgd_attack:
        validate_pgd(testloader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)
    '''
    train_time = time.time()

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


# Free Adversarial Training Module        
global global_noise_data
global_noise_data = torch.zeros([configs.DATA.batch_size, 3, configs.DATA.crop_size, configs.DATA.crop_size]).cuda()
def train(train_loader, model, criterion, optimizer, epoch):
    global global_noise_data
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = 0
    train_acc = 0
    train_err = 0
    train_n = 0
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.to(device)
        target = target.to(device)
        data_time.update(time.time() - end)
        for j in range(configs.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)
            loss = criterion(output, target)
            
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.5f} ({data_time.avg:.5f})\t'
                      'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
                sys.stdout.flush()
        train_loss += loss.item() * input.shape[0]
        train_acc += (output.max(1)[1] == target).sum().item()
        train_err += (output.max(1)[1] != target).sum().item()
        train_n += target.size(0)
    print("Accuracy: %.3f, Error: %.3f, Loss: %.3f" %(train_acc / len(train_loader.dataset), train_err / len(train_loader.dataset), train_loss / len(train_loader.dataset)))

if __name__ == '__main__':
    main()