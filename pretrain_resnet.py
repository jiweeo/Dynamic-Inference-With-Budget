'''
    pretrain the ResNet on CIFAR dataset
'''

import os
import torch
import torch.nn as nn
import torch.utils.data as D
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from utils import Data, Mode
import utils
import numpy as np


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Dynamic ResNet Training')
parser.add_argument('--lr', type=float, default=.1, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=80, #350,
        help='total epochs to run')
parser.add_argument('--data', type=str, default='c10')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

rootpath = utils.get_path(Data.mini_imagenet, iter=1)
os.makedirs(rootpath, exist_ok=True)

data_type = utils.str2Data(args.data)

def train(epoch):
    rnet.train()

    total = 0
    correct = 0
    train_loss = 0
    total_batch = 0

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        probs = rnet(inputs)
        optimizer.zero_grad()
        loss = criterion(probs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = probs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        total_batch += 1

    print('E:%d Train Loss: %.3f Train Acc: %.3f LR %f'
          % (epoch, 
             train_loss / total_batch, 
             correct / total, 
             optimizer.param_groups[0]['lr']))


def test(epoch):
    rnet.eval()

    total = 0
    correct = 0
    test_loss = 0
    total_batch = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            probs = rnet(inputs)
            loss = criterion(probs, targets)

            test_loss += loss.item()
            _, predicted = probs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            total_batch += 1

        print('E:%d Test Loss: %.3f Test Acc: %.3f'
              % (epoch, test_loss / total_batch, correct / total))

        # save best model
        acc = 100.*correct/total

        print('saving model...')
        state = {
            'net': rnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(rootpath, 'rnet.t7'))


def adjust_learning_rate(epoch, stage=list([40, 60])):
    order = np.sum(epoch >= np.array(stage))
    lr = args.lr * (0.1 ** order)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# dataset and dataloader
trainset, testset = utils.get_policy_datasets(Data.mini_imagenet, Mode.no_policy)
trainloader = D.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
testloader = D.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)


if data_type==Data.mini_imagenet:
    rnet = utils.get_resnet_model(data_type, [3, 4, 6, 3])  # resnet 50
else:
    rnet = utils.get_resnet_model(data_type, [18, 18, 18])  # resnet 110

rnet.to(device)
if torch.cuda.device_count() > 1:
    print('paralleling for multiple GPUs...')
    rnet = nn.DataParallel(rnet)

start_epoch = 0

if args.resume:
    assert os.path.isfile(os.path.join(rootpath, 'rnet.t7')), 'Error: no check-point found!'
    ckpt = torch.load(os.path.join(rootpath, 'rnet.t7'))
    rnet.load_state_dict(ckpt['net'])
    best_test_acc = ckpt['acc']
    start_epoch = ckpt['epoch']
else:
    # He's init
    for module in rnet.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

# Loss Fn and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

for epoch in range(start_epoch+1, args.max_epochs):
    train(epoch)
    test(epoch)
    adjust_learning_rate(epoch)
