#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, num_samples):
        """
        初始化数据集。
        
        参数:
            num_samples (int): 数据集中的样本数量。
        """
        self.num_samples = num_samples
        self.data = []
        self.labels = []

        # 生成数据和标签
        for _ in range(num_samples):
            # 生成一个形状为 (1, 28, 28) 的张量，均值为 0.1307，方差为 0.3081
            tensor = torch.randn(1, 28, 28) * np.sqrt(0.3081) + 0.1307
            # 生成一个范围在 1 到 10 之间的随机标签
            label = torch.randint(0, 10, (1,)).item()
            self.data.append(tensor)
            self.labels.append(label)
    
    def __len__(self):
        """返回数据集的长度。"""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        根据索引获取数据和标签。
        
        参数:
            idx (int): 索引值。
            
        返回:
            tuple: 包含张量和标签的元组。
        """
        return self.data[idx], self.labels[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv + bn
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8))
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        # depthwise conv
        self.depth_conv_bn1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, groups=16, bias=False),
            nn.BatchNorm2d(32))
        self.depth_conv_bn2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, groups=16, bias=False))
        # group conv
        self.group_conv1 = nn.Conv2d(48, 16, kernel_size=3, stride=2, groups=16, bias=True)
        self.group_conv2 = nn.Conv2d(48, 16, kernel_size=3, stride=1, groups=8, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=2, padding=1)

        self.fc_bn1 = nn.Sequential(
            nn.Linear(400, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        self.fc_bn2 = nn.Sequential(
            nn.Linear(128, 10, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True))

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.upsample = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, bias=False)


    def forward(self, x):
        x = self.conv_bn1(x)
        x = self.upsample(x)
        x = self.conv_bn2(x)
        x = self.avg_pool(x)
        x_1 = self.depth_conv_bn1(x)
        x_2 = self.depth_conv_bn2(x)
        x = torch.cat([x_1, x_2], 1) # c:16
        x_1 = self.group_conv1(x)
        x_2 = self.max_pool(x)
        x_2 = self.group_conv2(x_2)
        x = torch.add(x_1, x_2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # fc
        x = torch.flatten(x, 1)
        x = self.fc_bn1(x)
        x = self.dropout2(x)
        x = self.fc_bn2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST_AMCT Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    dataset1 = CustomDataset(6000)
    dataset2 = CustomDataset(200)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./model/mnist_cnn.pt")

    # save onnx
    random_input = torch.randn(1, 1, 28, 28, requires_grad=True)
    if use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        random_input = random_input.to(device)
    torch_out = torch.onnx._export(model,             # model being run
                                   random_input,                       # model input (or a tuple for multiple inputs)
                                   "./model/mnist_cnn.onnx", # where to save the model (can be a file or file-like object)
                                   export_params=True)      # store the trained parameter weights inside the model file

def test_model():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST_AMCT Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    dataset1 = CustomDataset(6000)
    dataset2 = CustomDataset(200)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load("./model/mnist_cnn.pt"))
    model.eval()

    test(model, device, test_loader)

if __name__ == '__main__':
    main()
    # test_model()
