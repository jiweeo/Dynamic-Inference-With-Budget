import torch.nn as nn
import torch
import torchvision.models as torchmodels
from torch.autograd import Variable
import torch.nn.functional as F
import math

from models import base
import utils

#--------------------------------------------------------------------------------------------------#
class FlatResNet(nn.Module):

    def seed(self, x):
        # x = self.relu(self.bn1(self.conv1(x))) -- CIFAR
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) -- ImageNet
        raise NotImplementedError

    # run a variable policy batch through the resnet implemented as a full mask over the residual
    # fast to train, non-indicative of time saving (use forward_single instead)
    def forward(self, x, policy=None):

        if policy is None:
            x = self.seed(x)

            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b == 0 else x
                    x = F.relu(residual + self.blocks[segment][b](x))

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            x = self.seed(x)

            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    action = policy[:,t].contiguous()
                    residual = self.ds[segment](x) if b==0 else x

                    # early termination if all actions in the batch are zero
                    if action.data.sum() == 0:
                        x = residual
                        t += 1
                        continue

                    action_mask = action.float().view(-1,1,1,1)
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    x = fx*action_mask + residual*(1-action_mask)
                    t += 1

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # run a single, fixed policy for all items in the batch
    # policy is a (15,) vector. Use with batch_size=1 for profiling
    def forward_single(self, x, policy):
        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
           for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                if policy[t]==1:
                    x = residual + self.blocks[segment][b](x)
                    x = F.relu(x)
                else:
                    x = residual
                t += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Smaller Flattened Resnet, tailored for CIFAR
class FlatResNet32(FlatResNet):

    def __init__(self, block, layers, num_classes=10):
        super().__init__()

        self.inplanes = 16
        self.conv1 = base.conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = base.DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample

# Regular Flattened Resnet, tailored for Imagenet etc.
class FlatResNet224(FlatResNet):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(FlatResNet224, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        strides = [1, 2, 2, 2]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.layer_config = layers

    def seed(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers, downsample


#---------------------------------------------------------------------------------------------------------#

class Policy32(nn.Module):

    def __init__(self, layer_config=[1, 1, 1], num_blocks=15):
        super(Policy32, self).__init__()
        self.features = FlatResNet32(base.BasicBlock, layer_config, num_classes=10)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()

        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, 1)
        self.limit = torch.cuda.FloatTensor([0])

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy32, self).load_state_dict(state_dict)

    def forward(self, x):
        x = self.features.forward_full(x)
        value = self.vnet(x)
        probs = F.sigmoid(self.logit(x))
        return probs, value

class Policy224(nn.Module):

    def __init__(self, layer_config=[1,1,1,1], num_blocks=16):
        super(Policy224, self).__init__()
        self.features = FlatResNet224(base.BasicBlock, layer_config, num_classes=1000)

        resnet18 = torchmodels.resnet18(pretrained=True)
        utils.load_weights_to_flatresnet(resnet18, self.features)

        self.features.avgpool = nn.AvgPool2d(4)
        self.feat_dim = self.features.fc.weight.data.shape[1]
        self.features.fc = nn.Sequential()


        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, 1)


    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy224, self).load_state_dict(state_dict)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = self.features.forward_full(x)
        value = self.vnet(x)
        probs = F.sigmoid(self.logit(x))
        return probs, value

