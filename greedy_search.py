from torch.autograd import Variable
import torch.utils.data as torchdata
import numpy as np
import tqdm
import pickle
import torch
import os
import torch.backends.cudnn as cudnn
from utils import Data, Mode
import utils
import argparse


def greedy_search(args, grouping_ratio):

    num_cuda = torch.cuda.device_count()
    data_type = utils.str2Data(args.data)

    # training set
    trainset, testset = utils.get_policy_datasets(Data.mini_imagenet, Mode.no_policy, augment_on_training=False)
    trainloader = torchdata.DataLoader(trainset, batch_size=num_cuda * args.batch_size, shuffle=False,
                                       num_workers=4, pin_memory=True)
    testloader = torchdata.DataLoader(testset, batch_size=num_cuda * args.batch_size, shuffle=False,
                                      num_workers=4, pin_memory=True)

    if args.test:
        loader = testloader
    else:
        loader = trainloader

    rnet = utils.get_resnet_model(data_type, [3, 4, 6, 3])  # resnet 50
    rnetpath = os.path.join(utils.get_path(Data.mini_imagenet, iter=args.iter), 'rnet.t7')
    rnet_dict = torch.load(rnetpath)
    rnet.load_state_dict(rnet_dict['net'])
    rnet.to(device)

    if num_cuda > 1:
        print('paralleling for multiple GPUs...')
        rnet = torch.nn.DataParallel(rnet)

    # total number of residual blocks in the ResNet
    num_block = sum(rnet.layer_config)
    rnet.eval().cuda(args.gpu)

    # the greedy policy is encoded as a vector of dropping order.
    # if greedy_search[image_id][i] = j, means the block j is the i-th block to prune.
    greedy_search = np.zeros((len(loader)*args.batch_size, num_block))
    print('Applying Greedy Search on %s dataset ...' % args.data)
    index_counter = 0
    print('Number of Layers: %d' % num_block)
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(loader), total=len(loader)):
        batch_size = inputs.shape[0]
        batch_policy = np.zeros((batch_size, num_block))

        # iteratively dropping residual blocks in the ResNet
        #   Limit is ranging from 1 to num_block, meaning the number of
        #   blocks has to be dropped.
        for limit in range(1, num_block+1):
            # initialize the max_prob with a negative values.
            max_prob = -100 * np.ones(batch_size)
            max_prob_drop_idx = np.zeros(batch_size).astype(np.int)
            drop_idx = np.zeros(batch_size).astype(np.int)

            for i in range(num_block-limit+1):
                # enumerate each remaining block, try to drop it
                # and then compute the prob
                for j in range(batch_size):
                    while batch_policy[j][drop_idx[j]] != 0:
                        drop_idx[j] += 1
                    batch_policy[j][drop_idx[j]] = limit
                p = torch.from_numpy((batch_policy == 0).astype(np.int)).view(batch_size, -1)
                p_var, inputs_var = Variable(p).cuda(args.gpu), Variable(inputs).cuda(args.gpu)
                preds = rnet.forward(inputs_var, p_var)
                for j in range(batch_size):
                    target = targets[j]
                    if preds.data[j][target] > max_prob[j]:
                        max_prob[j] = preds.data[j][target]
                    if preds.data[j][target] >= max_prob[j] * grouping_ratio:
                        max_prob_drop_idx[j] = drop_idx[j]
                    batch_policy[j][drop_idx[j]] = 0
                    drop_idx[j] += 1
            for j in range(batch_size):
                batch_policy[j][max_prob_drop_idx[j]] = limit
        greedy_search[index_counter: index_counter+batch_size] = batch_policy
        index_counter += batch_size

    writedir = utils.get_path(data_type.mini_imagenet, iter=args.iter)
    os.makedirs(writedir, exist_ok=True)
    if args.test:
        f = open(os.path.join(writedir, 'greedy_test.pkl'), 'wb')
    else:
        f = open(os.path.join(writedir, 'greedy_train.pkl'), 'wb')
    pickle.dump(greedy_search, f)
    print('done.')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Greedy optimal policy search')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--data', type=str, default='c10')
    args = parser.parse_args()

    # grouping ratio is used enhance the robustness of the greedy search method.
    greedy_search(args, 1.00)
