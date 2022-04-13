#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/9/7 22:04
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : MNISTConv.py 
# **************************************

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from RawModels.basic_module import BasicModule

# Training parameters for MNIST
MNIST_Training_Parameters = {
    'num_epochs': 20,
    'batch_size': 100,
    'learning_rate': 0.05,
    'momentum': 0.9,
    'decay': 1e-6
}
feature_out = None


def None_feature():
    global feature_out
    feature_out = None


def get_feature():
    return feature_out


def hook(module, fea_in, fea_out):
    global feature_out
    if feature_out is not None:
        feature_out = torch.cat((feature_out, fea_out.cpu()))
    else:
        feature_out = fea_out.cpu()


##orignal
class MNISTConvNet(BasicModule):
    def __init__(self, thermometer=False, level=1):
        super(MNISTConvNet, self).__init__()

        if thermometer is True:
            input_channels = 1 * level
        else:
            input_channels = 1

        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(4 * 4 * 64, 200)

        self.dropout = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(200, 200)
        self.fc2.register_forward_hook(hook)
        self.fc3 = nn.Linear(200, 10)
        self.middle = {}
        self.mid_layer = {}
        # softmax ? or not

    def forward(self, x):
        self.middle[0] = x.clone().detach().reshape((x.shape[0], -1))
        self.mid_layer[0] = x.clone().detach().reshape((x.shape[0], -1))
        out = self.conv32(x)
        self.middle[1] = out.clone().detach().reshape((out.shape[0], -1))

        out = self.conv64(out)
        self.middle[2] = out.clone().detach().reshape((out.shape[0], -1))
        # self.mid_cos[1] = out.clone().detach().reshape((out.shape[0], -1))


        out = out.view(-1, 4 * 4 * 64)

        out = F.relu(self.fc1(out))
        self.middle[3] = out.clone().detach().reshape((out.shape[0], -1))
        # self.mid_cos[2] = out.clone().detach().reshape((out.shape[0], -1))

        out = self.dropout(out)
        self.middle[4] = out.clone().detach().reshape((out.shape[0], -1))
        # self.mid_cos[3] = out.clone().detach().reshape((out.shape[0], -1))

        out = F.relu(self.fc2(out))
        self.middle[5] = out.clone().detach().reshape((out.shape[0], -1))

        out = self.fc3(out)
        self.middle[6] = out.clone().detach().reshape((out.shape[0], -1))
        self.mid_layer[0] = out.clone().detach().reshape((out.shape[0], -1))
        out = out - torch.max(out, dim=1, keepdim=True)[0]
        self.middle[7] = out.clone().detach().reshape((out.shape[0], -1))

        return out


# define the network architecture for MNIST-layel+1
# class MNISTConvNet(BasicModule):
#     def __init__(self, thermometer=False, level=1):
#         super(MNISTConvNet, self).__init__()
#
#         if thermometer is True:
#             input_channels = 1 * level
#         else:
#             input_channels = 1
#
#         self.conv32 = nn.Sequential(
#             nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv64 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.fc0 = nn.Linear(4 * 4 * 64, 512)
#         # self.fc01 = nn.Linear(512, 256)
#         # self.fc02 = nn.Linear(256, 512)
#         self.fc1 = nn.Linear(512, 200)
#         self.fc1.register_forward_hook(hook)
#         self.dropout = nn.Dropout2d(p=0.5)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc3 = nn.Linear(200, 10)
#         # softmax ? or not
#     def forward(self, x):
#         out = self.conv32(x)
#         out = self.conv64(out)
#         out = out.view(-1, 4*4*64)
#         out = F.relu(self.fc0(out))
#         # # out = F.relu(self.fc01(out))
#         # # out = F.relu(self.fc02(out))
#         out = F.relu(self.fc1(out))
#         out = self.dropout(out)
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
#         out = out - torch.max(out, dim=1, keepdim=True)[0]
#         return out

# conv3
# class MNISTConvNet(BasicModule):
#     def __init__(self, thermometer=False, level=1):
#         super(MNISTConvNet, self).__init__()
#
#         if thermometer is True:
#             input_channels = 1 * level
#         else:
#             input_channels = 1
#
#         self.conv32 = nn.Sequential(
#             nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv32_32 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
#             nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv64 = nn.Sequential(
#                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#                         nn.ReLU(),
#                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
#                         nn.ReLU(),
#                         nn.MaxPool2d(kernel_size=2)
#                     )
#
#         self.fc0 = nn.Linear(4 * 4 * 64, 512)
#         self.fc01 = nn.Linear(512, 256)
#         self.fc02 = nn.Linear(256, 512)
#         self.fc1 = nn.Linear(512, 200)
#         self.fc1.register_forward_hook(hook)
#         self.dropout = nn.Dropout2d(p=0.5)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc3 = nn.Linear(200, 10)
#     def forward(self, x):
#         out = self.conv32(x)
#         out=self.conv32_32(out)
#         out = self.conv64(out)
#         out = out.view(-1, 4*4*64)
#         out = F.relu(self.fc0(out))
#         # # out = F.relu(self.fc01(out))
#         # # out = F.relu(self.fc02(out))
#         out = F.relu(self.fc1(out))
#         out = self.dropout(out)
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
#         out = out - torch.max(out, dim=1, keepdim=True)[0]
#         return out

# layer+4
# class MNISTConvNet(BasicModule):
#     def __init__(self, thermometer=False, level=1):
#         super(MNISTConvNet, self).__init__()
#
#         if thermometer is True:
#             input_channels = 1 * level
#         else:
#             input_channels = 1
#
#         self.conv32 = nn.Sequential(
#             nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv64 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.fc0 = nn.Linear(4 * 4 * 64, 512)
#         self.fc01 = nn.Linear(512, 256)
#         self.fc02 = nn.Linear(256, 512)
#         self.fc1 = nn.Linear(512, 200)
#         self.fc1.register_forward_hook(hook)
#         self.dropout = nn.Dropout2d(p=0.5)
#         self.fc2 = nn.Linear(200, 200)
#         self.fc3 = nn.Linear(200, 10)
#         # softmax ? or not
#
#     def forward(self, x):
#         out = self.conv32(x)
#         # out = self.conv32_32(out)
#         out = self.conv64(out)
#         out = out.view(-1, 4 * 4 * 64)
#         out = F.relu(self.fc0(out))
#         out = F.relu(self.fc01(out))
#         out = F.relu(self.fc02(out))
#         out = F.relu(self.fc1(out))
#         out = self.dropout(out)
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
#         out = out - torch.max(out, dim=1, keepdim=True)[0]
#         return out


if __name__ == "__main__":
    model = MNISTConvNet()
    x = torch.randn(10, 1, 28, 28)
    out = model(x)
    # print(out.shape)
    # ct = 0
    # for child in model.children():
    #     ct += 1
    #     if ct < 8:
    #         print(child)
    feature=get_feature()
    print(feature)
