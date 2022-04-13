#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Time    : 2018/10/12 11:30
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : TrainTest.py 
# **************************************


import torch
import torch.nn.functional as F



# help functions for training and testing

def model_weight_decay(model=None):
    decay = None
    for (name, param) in model.named_parameters():
        if name.lower().find('conv') > 0:
            if decay is None:
                decay = param.norm(2)
            else:
                decay = decay + param.norm(2)
    if decay is None:
        decay = 0
    return decay

def contrastive_loss(label,representations):
    import torch
    import torch.nn.functional as F
    T = 0.1  # 温度参数T

    n = label.shape[0]  # batch

    # 假设我们的输入是5 * 3  5是batch，3是句

    # 这步得到它的相似度矩阵
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                            representations.unsqueeze(0),
                                            dim=2)
    # 这步得到它的label矩阵，相同label的位置为1
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(
        label.expand(n, n).t()))
    mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)
    similarity_matrix = similarity_matrix * mask_dui_jiao_0
    similarity_matrix = torch.exp(similarity_matrix / T)
    # 这步得到它的不同类的矩阵，不同类的位置为1
    loss_sum = 0
    for i in range(len(label)):
        # 这步得到它的不同类的矩阵，不同类的位置为1
        sim = (mask[i] * similarity_matrix[i]).sum()
        no_sim = similarity_matrix[i].sum()
        loss = torch.div(sim, no_sim)
        loss = -torch.log(loss)
        # loss = torch.log(sim)

        loss_sum += loss
    return loss_sum / len(label)
# train the model in one epoch
def train_one_epoch(model, train_loader, optimizer, epoch, device):
    """

    :param model:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param device:
    :return:
    """

    # Sets the model in training mode
    model.train()
    for index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # print(images.shape)
        # forward the nn
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\rTrain Epoch{:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
              format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, loss.item()), end=' ')
def train_one_epoch_by_NCE(model, train_loader, optimizer, epoch, device):
    """

    :param model:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param device:
    :return:
    """
    from ..MNISTConv import None_feature,get_feature
    # Sets the model in training mode
    model.train()
    for index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # forward the nn
        None_feature()
        outputs = model(images)
        feature=get_feature()
        loss = contrastive_loss(labels.cpu(),feature)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\rTrain Epoch{:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
              format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, loss.item()), end=' ')

def train_one_epoch_by_NCE_cifar(model, train_loader, optimizer, epoch, device):
    """

    :param model:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param device:
    :return:
    """
    from ..ResNet import None_feature,get_feature
    # Sets the model in training mode
    model.train()
    for index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # forward the nn
        None_feature()
        outputs = model(images)
        feature=get_feature()
        loss = contrastive_loss(labels.cpu(),feature)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\rTrain Epoch{:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
              format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, loss.item()), end=' ')

# evaluate the model using validation dataset
def validation_evaluation(model, validation_loader, device):
    """

    :param model:
    :param validation_loader:
    :param device:
    :return:
    """
    model = model.to(device)
    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
        ratio = correct / total
    print('validation dataset accuracy is {:.4f}'.format(ratio))
    return ratio


# evaluate the model using testing dataset
def testing_evaluation(model, test_loader, device):
    """

    :param model:
    :param test_loader:
    :param device:
    :return:
    """
    print('\n#####################################')
    print('#### The {} model is evaluated on the testing dataset loader ...... '.format(model.model_name))
    # Sets the module in evaluation mode.
    model = model.to(device)
    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
        ratio = correct / total
    print('#### Accuracy of the loaded model on the testing dataset: {:.1f}/{:.1f} = {:.2f}%'.format(correct, total, ratio * 100))
    print('#####################################\n')

    return ratio


def predict(model,dataloader,device):
    """
    :param model:
    :param dataloader:
    :param device:
    :return:
    """
    print('\n#####################################')
    print('#### The {} model is predict on the dataset loader ...... '.format(model.model_name))

    model=model.to(device)
    model.eval()
    predicts=None
    with torch.no_grad():
        for data,labels in dataloader:
            data,labels=data.to(device),labels.to(device)

            outputs=model(data)
            _,predicted=torch.max(outputs.data,1)
            if predicts is not None:
                predicts=torch.cat((predicts,predicted.cpu()))
            else :
                predicts=predicted.cpu()
    return predicts
