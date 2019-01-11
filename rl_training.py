'''
    Reinforcement Learning

'''
import os
import torch
import torch.backends.cudnn as cudnn
from utils import get_datasets, Mode, Data
import torch.utils.data as D
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Dynamic ResNet Reinforcement Learning')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
args = parser.parse_args()

trainset, testset = get_datasets(Data.cifar10, Mode.with_policy)
trainloader = D.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = D.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

