import torch.utils.data as torchdata
import numpy as np
import tqdm
import pickle
import torch
import os
import torch.backends.cudnn as cudnn
from utils import Data, Mode
import utils
from models.resnet_imagenet import resnet50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='scale search')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--data', type=str, default='c10')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--iter', type=int, default=1)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()


def scale_search():
    rnet.eval()
    scale_score = torch.zeros((len(loader)*args.batch_size, num_scales), dtype=torch.long)
    counter = 0
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(loader), total=len(loader)):
        batch_size = inputs.shape[0]
        inputs, targets = inputs.to(device), targets.to(device)
        for scale in range(num_scales):
            probs = rnet(inputs, scale=scale)
            _, predicted_cls = probs.max(1)
            mask = predicted_cls.eq(targets)
            for j in range(batch_size):
                if mask[j]:
                    scale_score[counter+j, scale] = 1
        counter += batch_size

    output_dir=''
    if dataset==Data.cifar10:
        output_dir = 'ckpt/Cifar10/ft%d/' % args.iter
    elif dataset==Data.imagenet:
        output_dir = 'ckpt/ImageNet/'

    os.makedirs(output_dir, exist_ok=True)
    if args.test:
        f = open(os.path.join(output_dir, 'scale_test.pkl'), 'wb')
    else:
        f = open(os.path.join(output_dir, 'scale_train.pkl'), 'wb')
    pickle.dump(scale_score, f)
    print('done.')


dataset = utils.str2Data(args.data)

# training set
trainset, testset = utils.get_scale_datasets(dataset, Mode.no_policy, augment_on_training=False)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=12)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=12)

if args.test:
    loader = testloader
else:
    loader = trainloader


if dataset == Data.imagenet:
    # rnet = utils.get_resnet_model(dataset, [])
    # rnet.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))
    rnet = resnet50(pretrained=True)

else:
    rnet = utils.get_resnet_model(dataset, [18, 18, 18])
    rnetpath = os.path.join(utils.get_path(Data.cifar10, iter=args.iter), 'rnet.t7')
    rnet_dict = torch.load(rnetpath)
    rnet.load_state_dict(rnet_dict['net'])

if dataset==Data.imagenet and device == 'cuda':
    rnet = torch.nn.DataParallel(rnet)

rnet.to(device)
num_scales = 3
scale_search()

