from step3_mydataset import MyDataset
from step4_net import cnn
import os
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

# ----------------------------------1：一些准备工作 路径以及数据
train_txt_path=os.path.join("data_tosplit","train.txt")
valid_txt_path=os.path.join("data_tosplit","valid.txt")

classes_name = ['cat','dog']

train_bs = 16
valid_bs = 16
lr_init = 0.001
max_epoch = 100

result_dir = os.path.join("result")
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

trainTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256, padding=4),
    transforms.ToTensor(),
])

validTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256, padding=4),
    transforms.ToTensor(),
])

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)


# ----------------------------------2：网络
net = cnn()     # 创建一个网络
net.initialize_weights()    # 初始化权值

# ----------------------------------3：损失函数和优化器
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

# ----------------------------------4：训练
for epoch in range(max_epoch):

    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for i, data in enumerate(train_loader):
        # enumerate参数为可遍历/可迭代的对象(如列表、字符串)
        # enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值，即需要index和value值的时候可以使用enumerate
        # if i == 30 : break
        # 获取图片和标签
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

# ----------------------------------5：验证集
    if epoch % 2 == 0:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        net.eval()
        for i, data in enumerate(valid_loader):

            # 获取图片和标签
            images, labels = data
            images, labels = Variable(images), Variable(labels)

            # forward
            outputs = net(images)
            outputs.detach_()

            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].numpy()
                pre_i = predicted[j].numpy()
                conf_mat[cate_i, pre_i] += 1.0

        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')

# ----------------------------------6：保存模型
net_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(net.state_dict(), net_save_path)
