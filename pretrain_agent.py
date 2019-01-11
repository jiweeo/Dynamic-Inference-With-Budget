'''
    pretrain the policy agent on CIFAR dataset
'''
import os
import torch
import torch.nn as nn
import torch.utils.data as D
import tqdm
import torch.optim as optim
from Networks import ResNet10
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import argparse
from utils import Data, Mode, get_policy_datasets

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Dynamic ResNet Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(epoch):
    agent.train()

    total = 0
    correct = 0
    train_loss = 0
    total_batch = 0

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        probs = agent(inputs)
        optimizer.zero_grad()
        loss = criterion(probs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = probs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        total_batch += 1

    print('E:%d Train Loss: %.3f Train Acc: %.3f'
          % (epoch, train_loss / total_batch, correct / total))


def test(epoch):
    global best_test_acc
    agent.eval()

    total = 0
    correct = 0
    test_loss = 0
    total_batch = 0

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        probs = agent(inputs)
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

    if acc > best_test_acc:
        best_test_acc = acc
        print('saving best model...')
        state = {
            'net': agent.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('policy_ckpt'):
            os.mkdir('policy_ckpt')
        torch.save(state, 'policy_ckpt/resnet10.t7')


def get_transform():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return train_tf, test_tf


# dataset and dataloader
trainset, testset = get_policy_dataset(Data.cifar10, Mode.no_policy)
trainloader = D.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = D.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
best_test_acc = 0.0

# agent network [2, 2, 2, 2]
agent = ResNet10(10)
agent.to(device)

# if device == 'cuda':
#     agent = torch.nn.DataParallel(agent)
#     cudnn.benchmark = True

start_epoch = 0

if args.resume:
    assert os.path.isdir('policy_ckpt'), 'Error: no such directory: ./policy_ckpt'
    ckpt = torch.load('policy_ckpt/resnet10.t7')
    agent.load_state_dict(ckpt['net'])
    best_test_acc = ckpt['acc']
    start_epoch = ckpt['epoch']

# Loss Fn and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(agent.parameters(), lr=args.lr)


for epoch in range(start_epoch+1, args.max_epochs):
    train(epoch)
    test(epoch)
