'''
    Imitation Learning
    Agent: Policy Network
    Expert: Greedy Optimal Policy
'''

import os
from tensorboard_logger import configure, log_value
import torch
import torch.utils.data as D
import tqdm
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import Networks
import torch.backends.cudnn as cudnn
from models import resnet, base
from Ranking_Loss import Ranking_Loss
import argparse
from utils import Data, Mode
import utils
import numpy as np

cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Dynamic ResNet Imitation Learning')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--iter', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--test', action='store_true')
parser.add_argument('--test_policy', action='store_true')
parser.add_argument('--data', type=str, default='c10')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--criterion', type=str, default='rank', help='rank, bce')
args = parser.parse_args()

dataset_type = utils.str2Data(args.data)
if dataset_type is None:
    raise AttributeError('Unknown data set!')

rootpath = utils.get_path(dataset_type, args.iter)

if not os.path.exists(rootpath):
    os.makedirs(rootpath, exist_ok=True)


def train(epoch, threshold):
    agent.train()
    rnet.eval()

    total = 0
    total_batch = 0
    sparsity = 0
    correct = 0
    train_loss = 0
    total_action = 0
    correct_action = 0
    for batch_idx, (inputs, targets, gt_policy) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        gt_policy = gt_policy.float()
        gt_policy = gt_policy.to(device)

        # predict policy
        probs = agent(inputs)
        optimizer.zero_grad()
        if args.criterion == 'bce':
            loss = criterion(probs, torch.max(gt_policy, 1)[1])
        else:
            loss = criterion(probs, gt_policy)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        sorted, _ = torch.sort(probs, 1)
        thresh = sorted[:, threshold]

        # compute the binary policy
        binary_policy = torch.zeros_like(probs)
        binary_policy[probs < thresh[:, None]] = 0.0
        binary_policy[probs >= thresh[:, None]] = 1.0
        binary_policy = binary_policy.to(device)

        # predict classification
        cls_preds = rnet.forward(inputs, binary_policy)
        _, predicted = cls_preds.max(1)

        # calculate metric: Accuracy and Sparsity
        correct += predicted.eq(targets).sum().item()
        sparsity += binary_policy.sum(1).mean()
        total += targets.shape[0]
        total_batch += 1

        # calculate corrected policy (according to the threshold)
        gt_binary_policy = torch.zeros_like(gt_policy)
        gt_binary_policy[gt_policy > threshold] = 1
        gt_binary_policy[gt_policy <= threshold] = 0
        correct_action += binary_policy.eq(gt_binary_policy).sum().item()
        total_action += gt_policy.shape[0] * gt_policy.shape[1]

    print('E:%d Train Loss: %.3f Train Acc: %.3f Sparsity: %.3f Precision: %.3f'
          % (epoch, train_loss / total_batch, correct / total, sparsity / total_batch, correct_action / total_action))
    log_value('train_acc', correct / total, epoch)
    log_value('train_loss', train_loss / total_batch, epoch)
    log_value('sparsity', sparsity / total_batch, epoch)


def test(epoch, threshold):
    agent.eval()
    rnet.eval()

    total = 0
    total_batch = 0
    sparsity = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, gt_policy) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            # predicted policy
            probs = agent(inputs)
            sorted, _ = torch.sort(probs, 1)
            thresh = sorted[:, threshold]

            # compute binary policy
            binary_policy = torch.zeros_like(probs)
            binary_policy[probs < thresh[:, None]] = 0.0
            binary_policy[probs >= thresh[:, None]] = 1.0
            binary_policy = binary_policy.to(device)

            # calculate metrics
            preds_map = rnet.forward(inputs, binary_policy)
            _, predicted = preds_map.max(1)
            correct += predicted.eq(targets).sum().item()
            sparsity += binary_policy.sum(1).mean()
            total += targets.shape[0]
            total_batch += 1

    print('E:%d Test Acc: %.3f Sparsity: %.3f' % (epoch, correct / total, sparsity / total_batch))
    log_value('test_acc', correct / total, epoch)
    log_value('test_sparsity', sparsity / total_batch, epoch)

    # save model
    acc = correct / total
    if not args.test:
        state = {
            'net': agent.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        torch.save(state, os.path.join(rootpath, 'agent_%d.t7' % epoch))
        torch.save(state, os.path.join(rootpath, 'agent_latest.t7'))


def test_policy(epoch, thresh):
    rnet.eval()

    total = 0
    total_batch = 0
    sparsity = 0
    correct = 0

    for batch_idx, (inputs, targets, gt_policy) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        # compute binary policy
        binary_policy = torch.zeros_like(gt_policy)
        binary_policy[gt_policy < thresh] = 0.0
        binary_policy[gt_policy >= thresh] = 1.0
        binary_policy = binary_policy.to(device)

        # calculate metrics
        preds_map = rnet.forward(inputs, binary_policy)
        _, predicted = preds_map.max(1)
        correct += predicted.eq(targets).sum().item()
        sparsity += binary_policy.sum(1).mean()
        total += targets.shape[0]
        total_batch += 1

    print('E:%d Test Acc: %.3f Sparsity: %.3f' % (epoch, correct / total, sparsity / total_batch))
    log_value('test_acc', correct / total, epoch)
    log_value('test_sparsity', sparsity / total_batch, epoch)


def adjust_learning_rate(epoch, stage=list([80, 120, 160])):
    order = np.sum(epoch >= np.array(stage))
    lr = args.lr * (0.1 ** order)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# data set and data loader
trainset, testset = utils.get_policy_datasets(dataset_type, Mode.with_policy, args.iter)
trainloader = D.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
testloader = D.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# global variable recording the best test accuracy
best_test_acc = 0.0

# configuration of the ResNet
rnet = utils.get_resnet_model(dataset_type, [3, 4, 6, 3])
rnet_dict = torch.load(os.path.join(rootpath, 'rnet.t7'))
rnet.load_state_dict(rnet_dict['net'])
num_blocks = sum(rnet.layer_config)

# Policy Network: ResNet-10
agent = Networks.PolicyNet10(num_blocks)
print(num_blocks)

# init weights of agent
start_epoch = 0
if args.resume:
    # test with trained check-point
    print('load the check point weights...')
    ckpt = torch.load(os.path.join(rootpath, 'agent_latest.t7'))
    start_epoch = ckpt['epoch']
    best_test_acc = ckpt['acc']
    agent.load_state_dict(ckpt['net'])
else:
    # train starting with pre-trained network
    print('train from scratch...')
    if dataset_type == Data.cifar10:
        pretrained_dict = torch.load(os.path.join('policy_ckpt', 'resnet10.t7'))
        model_dict = agent.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        agent.load_state_dict(model_dict)

# Loss Fn and Optimizer
if args.criterion == 'bce':
    criterion = CrossEntropyLoss()
elif args.criterion == 'rank':
    criterion = Ranking_Loss(num_blocks)
else:
    assert False
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# using gpu
agent.to(device)
rnet.to(device)

if args.test_policy:
    epoch = 0
    configure(os.path.join(rootpath, 'log/test_policy'), flush_secs=5)
    for limit in range(0, num_blocks):
        print('testing with %d blocks removed ...' % limit)
        epoch += 1
        test_policy(epoch, limit+1)

elif args.test:
    epoch = 0
    configure(os.path.join(rootpath, 'log/test_agent'), flush_secs=5)
    ckpt = torch.load(os.path.join(rootpath, 'agent_latest.t7'))
    agent.load_state_dict(ckpt['net'])
    for limit in range(0, num_blocks):
        print('testing with %d blocks removed ...' % limit)
        epoch += 1
        test(epoch, limit)
else:
    epoch = start_epoch
    configure(os.path.join(rootpath, 'log/train'), flush_secs=5)
    while epoch < start_epoch + args.max_epochs:
        epoch += 1
        threshold = 8
        train(epoch, threshold)
        test(epoch, threshold)
        adjust_learning_rate(epoch)
