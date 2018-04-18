# coding=utf-8
import os
import cv2
import numpy as np
import random

import torch
import torch.nn as nn
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# 训练参数
cuda = False
train_epoch = 1
train_lr = 0.005
train_momentum = 0.5
batchsize = 5
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# 测试训练集路径
test_path = "./img_6_Normalization/"
train_path = "./train/"

# 路径数据


def load_data(data_path):
    signal = os.listdir(data_path)
    for fsingal in signal:
        filepath = data_path + fsingal
        filename = os.listdir(filepath)
        for fname in filename:
            ffpath = filepath + "/" + fname
            path = [fsingal, ffpath]
            all_path.append(path)

        # 设立数据集多大
    count = len(all_path)
    data_x = np.empty((count, 1, 28, 28), dtype="float32")
    data_y = []
    # 打乱顺序
    random.shuffle(all_path)
    i = 0;

    # 读取图片　这里是灰度图　最后结果是i*i*i*i
    # 分别表示：batch大小　，　通道数，　像素矩阵
    for item in all_path:
        img = cv2.imread(item[1], 0)
        img = cv2.resize(img, (28, 28))
        arr = np.asarray(img, dtype="float32")
        data_x[i, :, :, :] = arr
        i += 1
        data_y.append(int(item[0]))

    data_x = data_x / 255
    data_y = np.asarray(data_y)
    #     lener = len(all_path)
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    dataset = dataf.TensorDataset(data_x, data_y)

    loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)

    return loader


#     print data_y

all_path = []
train_load = load_data(train_path)

all_path = []
test_load = load_data(test_path)


class L5_NET(nn.Module):
    def __init__(self):
        super(L5_NET, self).__init__();
        # 第一层输入1，20个卷积核　每个5*5
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        # 第二层输入20，30个卷积核　每个5*5
        self.conv2 = nn.Conv2d(20, 30, kernel_size=5)
        # drop函数
        self.conv2_drop = nn.Dropout2d()
        # 全链接层１，展开30*4*4，连接层50个神经元
        self.fc1 = nn.Linear(30 * 4 * 4, 50)
        # 全链接层１，50-4 ,4为最后的输出分类
        self.fc2 = nn.Linear(50, 10)

        # 前向传播

    def forward(self, x):
        # 池化层１ 对于第一层卷积池化，池化核2*2
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 池化层2 对于第二层卷积池化，池化核2*2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 平铺轴30*4*4个神经元
        x = x.view(-1, 30 * 4 * 4)
        # 全链接１
        x = F.relu(self.fc1(x))
        # dropout链接
        x = F.dropout(x, training=self.training)
        # 全链接w
        x = self.fc2(x)
        # softmax链接返回结果
        return F.log_softmax(x)


model = L5_NET()
if cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=train_lr, momentum=train_momentum)



# 预测函数
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_load):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # 求导
        optimizer.zero_grad()
        # 训练模型，输出结果
        output = model(data)
        # 在数据集上预测loss
        loss = F.nll_loss(output, target)
        # 反向传播调整参数pytorch直接可以用loss
        loss.backward()
        # SGD刷新进步
        optimizer.step()
        # 实时输出
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_load.dataset),
                       100. * batch_idx / len(train_load), loss.data[0]))

        #

'''
def train(epoch):
    model.train()#把module设成training模式，对Dropout和BatchNorm有影响
    for batch_idx, (data, target) in enumerate(train_load):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)#Variable类对Tensor对象进行封装，会保存该张量对应的梯度，以及对生成该张量的函数grad_fn的一个引用。如果该张量是用户创建的，grad_fn是None，称这样的Variable为叶子Variable。
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)#负log似然损失
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_load.dataset), 
                100. * batch_idx / len(train_load), loss.data[0]))
'''
# 测试函数
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_load:

        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        # 在测试集上预测
        output = model(data)
        # 计算在测试集上的loss
        test_loss += F.nll_loss(output, target).data[0]
        # 获得预测的结果
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        # 如果正确，correct+1
        correct += pred.eq(target.data).cpu().sum()

        # loss计算
    test_loss = test_loss
    test_loss /= len(test_load)
    # 输出结果
    print('\nThe {} epoch result : Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_load.dataset),
        100. * correct / len(test_load.dataset)))
    return correct, len(test_load.dataset),100. * correct / len(test_load.dataset)

def save_checkpoint(state, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)

def main():
    for epoch in range(1, train_epoch + 1):
        '''
        train(epoch)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict()
        })
        '''
        weights_file = './checkpoint.pth.tar'
        weights = torch.load(weights_file)['state_dict']
        model.load_state_dict(weights)
        correct,all,accuracy=test(epoch)
        return correct,all,accuracy

if __name__ == '__main__':
    main()