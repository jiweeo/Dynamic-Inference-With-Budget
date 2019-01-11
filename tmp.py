from models.resnet_imagenet import resnet50
model = resnet50(pretrained=True)
import torch
#out = model.forward(torch.FloatTensor(np
import numpy as np
out = model.forward(torch.FloatTensor(np.random.randn(1,3,224,224)))
out = model.forward(torch.FloatTensor(np.random.randn(1,3,112,112)))
%history -f tmp.py
