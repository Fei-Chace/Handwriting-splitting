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

# 训练参数
cuda = False
epoches = 20
learning_rate = 0.001
train_momentum = 0.5
batch_size = 5

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
    data_x = np.empty((count, 1, 32, 32), dtype="float32")
    data_y = []
    # 打乱顺序
    random.shuffle(all_path)
    i = 0;

    # 读取图片　这里是灰度图　最后结果是i*i*i*i
    # 分别表示：batch大小　，　通道数，　像素矩阵
    for item in all_path:
        img = cv2.imread(item[1], 0)
        img = cv2.resize(img, (32, 32))
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

    loader = dataf.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


#     print data_y

all_path = []
train_load = load_data(train_path)
all_path = []
test_load = load_data(test_path)


# build network
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


lenet = Lenet()
if cuda:
    lenet.cuda()

criterian = nn.CrossEntropyLoss(size_average=False)
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)

# 预测函数
lenet.train()
for i in range(epoches):
    running_loss = 0.
    running_acc = 0.
    for (img, label) in train_load:
        img = Variable(img)
        label = Variable(label)
        #print(img.size())
        optimizer.zero_grad()
        output = lenet(img)
        loss = criterian(output, label)
        # backward
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        running_acc += correct_num.data[0]

    running_loss /= len(train_load.dataset)
    running_acc /= len(train_load.dataset)
    print("[%d/%d] Loss: %.5f, Acc: %.2f" % (i + 1, epoches, running_loss, 100 * running_acc))
lenet.eval()

testloss = 0.
testacc = 0.
for (img, label) in test_load:
    img = Variable(img)
    label = Variable(label)

    output = lenet(img)
    loss = criterian(output, label)
    testloss += loss.data[0]
    _, predict = torch.max(output, 1)
    num_correct = (predict == label).sum()
    testacc += num_correct.data[0]

testloss /= len(test_load.dataset)
testacc /= len(test_load.dataset)
print("Test: %d Loss: %.5f, Acc: %.2f %%" % (len(test_load.dataset),testloss, 100 * testacc))










