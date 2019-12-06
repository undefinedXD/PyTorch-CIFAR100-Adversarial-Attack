# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# 不使用pytorch自己的图像预处理包， 使用时将下行取消注释
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import *
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from conf import global_settings
# Original utils
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

"""
Added By ChenWei Liu
"""
from tricks.XavierInit import init_weights
from tricks.NoWeightDecayOnBias import split_weights
from tricks.LabelSmoothing import LSR
# import transforms
# from new_utils import *

def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)
    # XavierInit初始化
    # net = init_weights(net)


    # 图片预处理 Added By Chenwei Liu
    # cifar100_train_transforms = transforms.Compose([
    #     transforms.ToCVImage(),
    #     transforms.RandomResizedCrop(32), # ImageSize
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
    #     transforms.RandomErasing(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(global_settings.CIFAR100_TRAIN_MEAN, global_settings.CIFAR100_TRAIN_STD)
    # ])

    # cifar100_test_transforms = transforms.Compose([
    #     transforms.ToCVImage(),
    #     transforms.CenterCrop(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize(global_settings.CIFAR100_TRAIN_MEAN, global_settings.CIFAR100_TRAIN_STD)
    # ])

    # cifar100_training_loader = new_get_train_dataloader(
    #     global_settings.CIFAR100_PATH,
    #     cifar100_train_transforms,
    #     num_workers=args.w,
    #     batch_size=args.b,
    # )

    # cifar100_testing_loader = new_get_test_dataloader(
    #     global_settings.CIFAR100_PATH,
    #     cifar100_test_transforms,
    #     num_workers=args.w,
    #     batch_size=args.b,
    # )

    # data preprocessing Base on PyTorch:

    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    loss_function = nn.CrossEntropyLoss()
    
    # 引入LabelSmooth
    # loss_function = LSR()


    # 引入No Weight Decay On Bias方法
    # params = split_weights(net)
    # optimizer = optim.SGD(params,lr=args.lr,momentum=0.9,weight_decay=1e-4,nesterov=True)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) #不采用no weight decay on bias优化
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
