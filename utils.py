import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from enum import Enum
import cifar_policy, cifar_scale
from models import resnet, base
from imagenet_scale import ImageNetWithScale
from imagenet_policy import ImageNetWithPolicy

class Data(Enum):
    cifar10 = 1
    cifar100 = 2
    imagenet = 3
    mini_imagenet = 4


class Mode(Enum):
    with_policy = 1
    no_policy = 0


def get_transform(dataset, augment=True):
    if dataset == Data.cifar10:
        if augment:
            tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    elif dataset == Data.cifar100:
        if augment:
            tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    elif dataset == Data.imagenet:
        if augment:
            tf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    elif dataset == Data.mini_imagenet:
        if augment:
            tf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    else:
        tf = None
    return tf


def get_policy_datasets(dataset, mode, augment_on_training=True, iter=1):
    train_tf = get_transform(dataset, augment=augment_on_training)
    test_tf = get_transform(dataset, augment=False)

    if dataset == Data.cifar10:
        if mode == Mode.with_policy:
            train_set = cifar_policy.CIFAR10(root='./data', train=True, download=True, transform=train_tf, iter=iter)
            test_set = cifar_policy.CIFAR10(root='./data', train=False, download=True, transform=test_tf, iter=iter)
        else:
            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
    elif dataset == Data.cifar100:
        train_set = None
        test_set = None
    elif dataset == Data.imagenet:
        train_set = None
        test_set = None
    elif dataset == Data.mini_imagenet:
        if mode == Mode.with_policy:
            train_set = ImageNetWithPolicy('./mini_ImageNet/train',
                                           os.path.join(get_path(dataset, iter), 'greedy_train.pkl'),
                                           transform=train_tf)
            test_set = ImageNetWithPolicy('./mini_ImageNet/val',
                                          os.path.join(get_path(dataset, iter), 'greedy_test.pkl'),
                                          transform=test_tf)
        else:
            train_set = datasets.ImageFolder('./mini_ImageNet/train', transform=train_tf)
            test_set = datasets.ImageFolder('./mini_ImageNet/val', transform=test_tf)
    else:
        assert 0
    return train_set, test_set


def get_scale_datasets(dataset, mode, augment_on_training=True, iter=1):
    train_tf = get_transform(dataset, augment=augment_on_training)
    test_tf = get_transform(dataset, augment=False)

    if dataset == Data.cifar10:
        if mode == Mode.with_policy:
            train_set = cifar_scale.CIFAR10(root='./data', train=True, download=True, transform=train_tf, iter=iter)
            test_set = cifar_scale.CIFAR10(root='./data', train=False, download=True, transform=test_tf, iter=iter)
        else:
            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
    elif dataset == Data.cifar100:
        train_set = None
        test_set = None
    elif dataset == Data.imagenet:
        if mode == Mode.no_policy:
            train_set = datasets.ImageFolder('/media/hdd1/datasets/mini_ImageNet/train', transform=train_tf)
            test_set = datasets.ImageFolder('/media/hdd1/datasets/mini_ImageNet/val', transform=test_tf)
        else:
            train_set = ImageNetWithScale('/media/hdd1/datasets/mini_ImageNet/train', 'ckpt/ImageNet/scale_train.pkl', transform=train_tf)
            test_set = ImageNetWithScale('/media/hdd1/datasets/mini_ImageNet/val', 'ckpt/ImageNet/scale_test.pkl', transform=test_tf)
    else:
        assert 0
    return train_set, test_set


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_path(data, iter):
    if data==Data.cifar10:
        return os.path.join('ckpt', 'Cifar10', 'ft%d' %iter)
    elif data==Data.cifar100:
        return os.path.join('ckpt', 'Cifar100', 'ft%d' %iter)
    elif data==Data.mini_imagenet:
        return os.path.join('ckpt', 'mini_ImageNet', 'ft%d' % iter)
    else:
        return os.path.join('ckpt', 'ImageNet', 'ft%d' %iter)


def str2Data(str):
    if str=='c10' or str == 'Cifar10' or str== 'CIFAR10':
        return Data.cifar10
    elif str=='c100' or str == 'Cifar100' or str== 'CIFAR100':
        return Data.cifar100
    elif str=='imagenet' or str == 'ImageNet' or str== 'IMAGENET':
        return Data.imagenet
    elif str=='mini' or str == 'mini_imagenet':
        return Data.mini_imagenet
    else:
        return None


def get_resnet_model(dataset_type, blocks_list=[18, 18, 18]):
    if dataset_type==Data.cifar10:
        return resnet.FlatResNet32(base.BasicBlock, blocks_list, num_classes=10)
    elif dataset_type==Data.cifar100:
        return resnet.FlatResNet32(base.BasicBlock, blocks_list, num_classes=100)
    elif dataset_type == Data.mini_imagenet:
        return resnet.FlatResNet224(base.Bottleneck, blocks_list, num_classes=80)
    else:
        return resnet.FlatResNet224(base.Bottleneck, blocks_list, num_classes=1000)
