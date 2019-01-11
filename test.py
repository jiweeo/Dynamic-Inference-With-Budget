import torch
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import numpy as np
import tqdm
import utils
from tensorboard_logger import configure, log_value
from torchvision import transforms
import cifar_policy
import torch.backends.cudnn as cudnn
import os

cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='R110_C10')
parser.add_argument('--data_dir', default='data/')
parser.add_argument('--load', default=None)
args = parser.parse_args()

#---------------------------------------------------------------------------------------------#
class FConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FConv2d, self).forward(x)
        output_area = output.size(-1)*output.size(-2)
        filter_area = np.prod(self.kernel_size)
        self.num_ops += 2*self.in_channels*self.out_channels*filter_area*output_area
        return output

class FLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FLinear, self).__init__(in_features, out_features, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FLinear, self).forward(x)
        self.num_ops += 2*self.in_features*self.out_features
        return output

def count_flops(model, reset=True):
    op_count = 0
    for m in model.modules():
        if hasattr(m, 'num_ops'):
            op_count += m.num_ops
            if reset: # count and reset to 0
                m.num_ops = 0

    return op_count

# replace all nn.Conv and nn.Linear layers with layers that count flops
nn.Conv2d = FConv2d
nn.Linear = FLinear

#--------------------------------------------------------------------------------------------#

def test(limit):

    total_ops = []
    matches, policies = [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs, targets = Variable(inputs, volatile=True).cuda(), Variable(targets).cuda()
        probs, _ = agent(inputs)

        _, order = torch.sort(probs, 1)

        policy = probs.clone()
        policy[order < limit] = 0.0
        policy[order >= limit] = 1.0

        preds = rnet.forward(inputs, policy)
        _ , pred_idx = preds.max(1)
        match = (pred_idx == targets).data.float()

        matches.append(match)
        policies.append(policy.data)

        ops = count_flops(agent) + count_flops(rnet)
        total_ops.append(ops)

    accuracy, _, sparsity, variance, policy_set = utils.performance_stats(policies, matches, matches)
    ops_mean, ops_std = np.mean(total_ops), np.std(total_ops)

    log_str = u'''
    Accuracy: %.3f
    Block Usage: %.3f \u00B1 %.3f
    FLOPs/img: %.2E \u00B1 %.2E
    Unique Policies: %d
    '''%(accuracy, sparsity, variance, ops_mean, ops_std, len(policy_set))
    log_value('test_acc', accuracy, limit)
    print(log_str)

#--------------------------------------------------------------------------------------------------------#
def get_transform():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform_train, transform_test


# dataset and dataloader
transform_train, transform_test = get_transform()
trainset = cifar_policy.CIFAR10(root='./data', train=True, download=True, transform= transform_train)
_, testset = utils.get_dataset(args.model, args.data_dir)
testloader = torchdata.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
trainloader = torchdata.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
rnet, agent = utils.get_model(args.model)

# if no model is loaded, use all blocks
agent.logit.weight.data.fill_(0)
agent.logit.bias.data.fill_(10)

print("loading checkpoints")

if args.load is not None:
    utils.load_checkpoint(rnet, agent, args.load)

rnet.eval().cuda()
agent.eval().cuda()

configure('cv/R110_C10_sl/BlockDrop_test')

for limit in range(0, 54):
    print('------------------------%d-------------------' %limit)
    test(limit)
