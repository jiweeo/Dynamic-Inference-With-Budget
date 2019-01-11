import os
from tensorboard_logger import configure, log_value
import torch
import torch.utils.data as D
import tqdm
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCELoss
import Networks
import torch.backends.cudnn as cudnn
from models import resnet, base, resnet_imagenet
import argparse
import utils
import numpy as np

cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Learning Scale')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--iter', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=60, help='total epochs to run')
parser.add_argument('--test', action='store_true')
parser.add_argument('--test_policy', action='store_true')
parser.add_argument('--data', type=str, default='c10')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

dataset_type = utils.str2Data(args.data)
if dataset_type is None:
    raise AttributeError('Unknown data set!')

rootpath = utils.get_path(dataset_type, args.iter)

if not os.path.exists(rootpath):
    os.makedirs(rootpath, exist_ok=True)


def train(epoch):
    agent.train()
    rnet.eval()

    total = 0
    total_batch = 0
    correct = np.zeros(num_scales)
    train_loss = 0

    for batch_idx, (inputs, targets, gt_scale) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        gt_scale = gt_scale.to(device)
        probs = agent(inputs)
        optimizer.zero_grad()
        loss = criterion(probs, gt_scale.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        hard_pred = probs > torch.Tensor([0.5]).to(device)
        hard_pred = hard_pred.long()
        batch_corrent = torch.sum(hard_pred, dim=0).cpu().numpy()
        correct += batch_corrent
        total += targets.shape[0]
        total_batch += 1

        if batch_idx % 10 == 0:
            print('E:%d Train Loss: %.3f Train Scale Acc: %.3f Train Scale Loss(current): %.3f '
                  % (epoch, train_loss / total_batch, correct.sum() / (total * num_scales), loss))
            print(batch_corrent / inputs.shape[0])
    # log_value('train_acc', correct / total, epoch)
    # log_value('train_loss', train_loss / total_batch, epoch)


def test(epoch):
    agent.eval()
    rnet.eval()

    global best_test_acc
    total = 0
    corrected_cls = 0
    corrected_scale = 0
    usage = 0
    t = torch.Tensor([0.5]).to(device)

    scale_count = np.zeros(num_scales)
    for batch_idx, (inputs, targets, gt_scale) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets, gt_scale= inputs.to(device), targets.to(device), gt_scale.to(device)

        # predicted policy
        probs = agent(inputs)
        hard_preds = probs > torch.Tensor([0.5]).to(device)
        hard_preds = hard_preds.long()

        corrected_scale += hard_preds.eq(gt_scale).sum().item()
        scale_count += sum(hard_preds, 0).cpu().numpy()

        single_scale_preds = rnet(inputs, 0)
        preds_map = single_scale_preds
        for scale in range(num_scales + 1):
            single_scale_preds = rnet(inputs, scale=scale)

            mask = hard_preds[:, scale]
            preds_map[mask.nonzero(), :] = single_scale_preds[mask.nonzero(), :]
            count = float(sum(mask).item())
            usage += count / (scale + 1) / (scale + 1) - count

        _, predicted_cls = preds_map.max(1)
        corrected_cls += predicted_cls.eq(targets).sum().item()
        total += targets.shape[0]

    print('E:%d Test Class Acc: %.3f Test Scale Acc: %.3f Usage: %.3f'
          % (epoch, corrected_cls / total, corrected_scale / (total * num_scales), (usage+total) / total))
    print(scale_count / total)
    # log_value('test_acc', corrected_cls / total, epoch)
    # log_value('test_sparsity', sparsity / total, epoch)

    # save model
    if not args.test:
        if corrected_scale / total > best_test_acc:
            print('saving best model...')
            best_test_acc = corrected_scale / total
            state = {
                'net': agent.state_dict(),
                'acc': best_test_acc,
                'epoch': epoch
            }
            torch.save(state, os.path.join(rootpath, 'scale_best.t7'))

        state = {
            'net': agent.state_dict(),
            'acc': best_test_acc,
            'epoch': epoch
        }
        torch.save(state, os.path.join(rootpath, 'scale_latest.t7'))


def test_policy(epoch):
    rnet.eval()

    total = 0
    total_batch = 0
    sparsity = 0
    correct = 0

    for batch_idx, (inputs, targets, gt_scale) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate metrics
        if dataset_type==utils.Data.imagenet:
            preds_map = torch.zeros(inputs.shape[0], 1000).to(device)
        else:
            preds_map = torch.zeros(inputs.shape[0], 10).to(device)
        for scale in range(num_scales):
            single_scale_pred = rnet(inputs, scale=scale)
            mask = gt_scale == scale
            preds_map[mask, :] = single_scale_pred[mask, :]

        _, predicted = preds_map.max(1)
        correct += predicted.eq(targets).sum().item()

        sparsity += gt_scale.float().mean()
        total += targets.shape[0]
        total_batch += 1

    print('E:%d Test Acc: %.3f Sparsity: %.3f' % (epoch, correct / total, sparsity / total_batch))
    # log_value('test_acc', correct / total, epoch)
    # log_value('test_sparsity', sparsity / total_batch, epoch)


def adjust_learning_rate(epoch, stage=[20, 40]):
    order = np.sum(epoch >= np.array(stage))
    lr = args.lr * (0.1 ** order)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# data set and data loader
train_tf = utils.get_transform(dataset_type, augment=True)
test_tf = utils.get_transform(dataset_type, augment=False)


trainset, testset = utils.get_scale_datasets(dataset_type, utils.Mode.with_policy)
trainloader = D.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
testloader = D.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

# global variable recording the best test accuracy
best_test_acc = 0.0

start_epoch = 0

if dataset_type == utils.Data.imagenet:
    num_scales = 2
    pretrained_agent = resnet_imagenet.resnet18(pretrained=True)
    # agent = resnet_imagenet.ResNet(resnet_imagenet.BasicBlock, [2, 2, 2, 2], num_scales)
    agent = resnet_imagenet.scalenet18(pretrained=True, scalelist=[1, 0.5, 0.25])

    # pretrained_dict = pretrained_agent.state_dict()
    # model_dict = agent.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}
    # model_dict.update(pretrained_dict)
    # agent.load_state_dict(model_dict)

    rnet = resnet_imagenet.resnet50(pretrained=True)

else:
    # ScaleNet
    num_scales = 2
    agent = Networks.ScaleNet10(num_scales)

    # configuration of the ResNet
    rnet = resnet.FlatResNet32(base.BasicBlock, [18, 18, 18], num_classes=10)
    rnet_dict = torch.load(os.path.join(rootpath, 'rnet.t7'))
    rnet.load_state_dict(rnet_dict['net'])

    if args.resume:
        # test with trained check-point
        print('load the check point weights...')
        ckpt = torch.load(os.path.join(rootpath, 'scale_latest.t7'))
        start_epoch = ckpt['epoch']
        best_test_acc = ckpt['acc']
        agent.load_state_dict(ckpt['net'])
    else:
        # train starting with pre-trained network
        print('train from scratch...')
        pretrained_dict = torch.load(os.path.join('policy_ckpt', 'resnet10.t7'))
        model_dict = agent.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        agent.load_state_dict(model_dict)

criterion = BCELoss()
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# only parallelize for imagenet
if dataset_type == utils.Data.imagenet and device == 'cuda':
    agent = torch.nn.DataParallel(agent)
    rnet = torch.nn.DataParallel(rnet)

# using gpu
agent.to(device)
rnet.to(device)

if args.test_policy:
    epoch = 0
    configure(os.path.join(rootpath, 'log/test_policy'), flush_secs=5)
    test_policy(epoch)
elif args.test:
    epoch = 0
    configure(os.path.join(rootpath, 'log/scale_latest.t7'), flush_secs=5)
    ckpt = torch.load(os.path.join(rootpath, 'scale_latest.t7'))
    agent.load_state_dict(ckpt['net'])
    test(epoch)
else:
    epoch = start_epoch
    configure(os.path.join(rootpath, 'log/scale'), flush_secs=5)
    while epoch < start_epoch + args.max_epochs:
        train(epoch)
        test(epoch)
        adjust_learning_rate(epoch)
        epoch += 1
