import os
from tensorboard_logger import configure, log_value
import torch
import torch.utils.data as D
import numpy as np
import tqdm
from utils import get_policy_datasets, Data, Mode
import torch.optim as optim
import Networks
import torch.backends.cudnn as cudnn
from models import resnet, base
import argparse
from utils import Data, get_path

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Dynamic ResNet Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--iter', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=160, help='total epochs to run')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

rootpath = get_path(Data.cifar10, args.iter)
savepath = get_path(Data.cifar10, args.iter+1)
os.makedirs(savepath, exist_ok=True)


def train(epoch):
    agent.eval()
    rnet.train()

    total = 0
    correct = 0

    sample_props = np.square(np.arange(54))[::-1]
    sample_props = sample_props / sample_props.sum()

    for batch_idx, (inputs, targets, gt_policy) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        # predict policy
        probs = agent(inputs)

        # random a time budget
        limit = np.random.choice(54, p=sample_props)
        # limit = 0

        # convert to binary actions
        sorted, _ = torch.sort(probs, 1)
        thresh = sorted[:, limit]
        binary_policy = torch.zeros_like(probs)
        binary_policy[probs < thresh[:, None]] = 0.0
        binary_policy[probs >= thresh[:, None]] = 1.0
        binary_policy = binary_policy.to(device)

        # use gt_policy to predict label
        cls_preds = rnet.forward(inputs, binary_policy)
        _, predicted = torch.max(cls_preds, 1)

        # update rnet
        rnet_optimizer.zero_grad()
        loss = rnet_criteron(cls_preds, targets)
        loss.backward()
        rnet_optimizer.step()

        # calculate metric: Accuracy and Sparsity
        correct += predicted.eq(targets).cpu().sum().item()
        total += targets.shape[0]

    print('E%d: Train Acc: %.3f' % (epoch,  correct/total))
    log_value('train_acc', correct/total, epoch)


def test(epoch, limit):
    global acc
    agent.eval()
    rnet.eval()

    total = 0
    correct = 0

    for batch_idx, (inputs, targets, gt_policy) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        # predicted policy
        probs = agent(inputs)
        sorted, _ = torch.sort(probs, 1)
        thresh = sorted[:, limit]

        binary_policy = torch.zeros_like(probs)
        binary_policy[probs < thresh[:, None]] = 0.0
        binary_policy[probs >= thresh[:, None]] = 1.0
        binary_policy = binary_policy.to(device)

        # calculate metrics
        preds_map = rnet.forward(inputs, binary_policy)
        _, predicted = torch.max(preds_map, 1)
        correct += predicted.eq(targets).cpu().sum().item()

        total += targets.shape[0]

    print('E:%d Test Acc: %.3f' % (epoch, correct / total))
    acc = acc + (correct/total)


# dataset and dataloader
trainset, testset = get_policy_datasets(Data.cifar10, Mode.with_policy, iter=args.iter)
trainloader = D.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = D.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

# global metric
best_test_acc = 0.0
epoch = 0

# configuration of the ResNet
if args.resume:
    rnet = resnet.FlatResNet32(base.BasicBlock, [18, 18, 18], num_classes=10)
    state = torch.load(os.path.join(savepath, 'rnet.t7'))
    rnet.load_state_dict(state['net'])
    best_test_acc = state['acc']
    epoch = state['epoch']
else:
    rnet = resnet.FlatResNet32(base.BasicBlock, [18, 18, 18], num_classes=10)
    rnet_dict = torch.load(os.path.join(rootpath, 'rnet.t7'))
    rnet.load_state_dict(rnet_dict['net'])

# Policy Network: ResNet-10
agent = Networks.PolicyNet10()
ckpt_dict = torch.load(os.path.join(rootpath, 'agent_best.t7'))
agent.load_state_dict(ckpt_dict['net'])

# Loss Fn and Optimizer
rnet_criteron = torch.nn.CrossEntropyLoss()
rnet_optimizer = optim.Adam(rnet.parameters(), lr=args.lr)

# using gpu
agent.to(device)
rnet.cuda()

acc = 0.0
configure(rootpath+'/log', flush_secs=5)
while epoch < args.max_epochs:
    epoch += 1
    train(epoch)
    if epoch % 1 == 0:
        acc = 0.0
        test(epoch, 0)
        test(epoch, 27)
        test(epoch, 34)
        acc = acc / 3
        log_value('test_acc', acc, epoch)
        if acc>best_test_acc:
            # save model
            best_test_acc = acc
            state = {
                'net': rnet.state_dict(),
                'acc': best_test_acc,
                'epoch': epoch
            }
            torch.save(state, os.path.join(savepath, 'rnet.t7'))

