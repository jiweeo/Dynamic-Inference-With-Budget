import os
from tensorboard_logger import configure, log_value
import torch
import torch.utils.data as D
import tqdm
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import Networks
import torch.backends.cudnn as cudnn
from models import resnet, base, resnet_imagenet
import argparse
import utils
import numpy as np

cudnn.benchmark = True
torch.cuda.manual_seed(7)
torch.manual_seed(7)
np.random.seed(7)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--iter', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=60, help='total epochs to run')
parser.add_argument('--test', action='store_true')
parser.add_argument('--data', type=str, default='c10')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

dataset_type = utils.str2Data(args.data)
if dataset_type is None:
    raise AttributeError('Unknown data set!')

rootpath = utils.get_path(dataset_type, args.iter)

if not os.path.exists(rootpath):
    os.makedirs(rootpath, exist_ok=True)


def get_reward(scales, corrected, gamma=-1):
    r = torch.zeros_like(scales)
    mask = corrected == 1
    gains = 1.5 - torch.pow(scales, 2)
    r[mask.nonzero()] = gains[mask.nonzero()]  # gains
    r[1-mask] = gamma  # penalty
    return r


def predict_class(inputs, targets, scales_idx, num_scales):

    out = torch.zeros(inputs.shape[0], 1000).to(device)
    for scale in range(num_scales):
        single_out = rnet(inputs, scale)
        mask = scales_idx == torch.Tensor([scale]).long().to(device)
        out[mask.nonzero(), :] = single_out[mask.nonzero(), :]

    loss = criterionRnet(out, targets)
    _, pred_cls = out.max(1)

    return loss, pred_cls


def get_usage(scales_idx, scalelist):
    usage = 0
    scalearray = scales_idx.cpu().numpy()
    for i in range(len(scalelist)):
        count = np.sum(scalearray == i)
        usage += count * scalelist[i]**2
    return usage


def train(epoch):
    print()
    agent.train()
    rnet.train()

    total = 0
    total_batch = 0
    total_correct = 0
    total_rnet_loss = 0
    total_agent_loss = 0
    epsilon = 0.1
    usage = 0

    scalelistGPU = torch.Tensor(scalelist).to(device)
    for batch_idx, (inputs, targets, _) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        # sample actions (eps-exploration)
        rand_num = np.random.uniform()
        probs = agent(inputs)
        if rand_num > epsilon:
            scales_idx = torch.multinomial(probs, num_samples=1)
        else:
            scales_idx = torch.randint(high=len(scalelist), size=(inputs.shape[0], 1),
                                       dtype=torch.long, device=device)
        scales = scalelistGPU[scales_idx]
        p = torch.gather(probs, 1, scales_idx)
        # update rnet
        rnet_loss, pred_cls = predict_class(inputs, targets, scales_idx, num_scales=len(scalelist))
        optimizerRnet.zero_grad()
        rnet_loss.backward()
        optimizerRnet.step()
        # update agent
        correct = pred_cls.eq(targets)
        agent_loss = torch.sum(-p * get_reward(scales, correct))
        optimizerAgent.zero_grad()
        agent_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 10)
        optimizerAgent.step()
        # metrics
        total_batch += 1
        total_correct += correct.sum().item()
        total += inputs.shape[0]
        total_rnet_loss += rnet_loss.item()
        total_agent_loss += agent_loss.item()
        usage += get_usage(scales_idx, scalelist)

        if batch_idx % 10 == 0:
            print('[Train E:%d Rnet Loss: %.3f  AgentLoss: %.3f ACC: %.3f Usage: %.3f'
                  % (epoch, total_rnet_loss / total_batch, total_agent_loss / total_batch,
                     total_correct/total, usage/total))
    # log_value('train_acc', correct / total, epoch)
    # log_value('train_loss', train_loss / total_batch, epoch)


def test(epoch):
    agent.eval()
    rnet.eval()
    global best_test_acc

    total = 0
    total_batch = 0
    total_loss = 0
    total_correct = 0
    usage = 0
    for batch_idx, (inputs, targets, _) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        inputs, targets = inputs.to(device), targets.to(device)

        # predicted policy
        probs = agent(inputs)
        _, pred_scale = probs.max(1)  # greedy
        loss, pred_cls = predict_class(inputs, targets, pred_scale, len(scalelist))
        total_loss += loss.item()
        total_correct += pred_cls.eq(targets).sum().item()
        total += inputs.shape[0]
        total_batch += 1
        usage += get_usage(pred_scale, scalelist)

    print('[Test] E:%d Rnet Loss: %.3f Test Class Acc: %.3f Usage: %.3f'
          % (epoch,  total_loss/total_batch, total_correct / total, usage / total))

    # save model
    if not args.test:
        if total_correct / total > best_test_acc:
            print('saving best model...')
            best_test_acc = total_correct / total
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
        torch.save(state, os.path.join(rootpath, 'agent_E_%d_%.3f.t7' % (epoch, total_correct/total)))


def adjust_learning_rate(epoch, stage=(20, 40)):
    order = np.sum(epoch >= np.array(stage))
    lr = args.lr * (0.1 ** order)
    for param_group in optimizerAgent.param_groups:
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
    scalelist = [1, 0.5, 0.25]
    pretrained_agent = resnet_imagenet.resnet18(pretrained=True)
    agent = resnet_imagenet.scalenet18(pretrained=True, scalelist=scalelist)
    rnet = resnet_imagenet.resnet50(pretrained=True)

else:
    # ScaleNet
    scalelist = [1, 0.5]
    agent = Networks.ScaleNet10(len(scalelist))

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

criterionRnet = CrossEntropyLoss()
optimizerAgent = optim.Adam(agent.parameters(), lr=args.lr)
optimizerRnet = optim.Adam(agent.parameters(), lr=1e-5)

# only parallelize for imagenet
if dataset_type == utils.Data.imagenet and device == 'cuda':
    agent = torch.nn.DataParallel(agent)
    rnet = torch.nn.DataParallel(rnet)

# using gpu
agent.to(device)
rnet.to(device)


if args.test:
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
