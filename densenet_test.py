from torchvision import models
from RawModels.Utils.dataset import get_cifar10_test_loader,get_cifar10_train_validate_loader
from RawModels.ResNet import CIFAR10_Training_Parameters
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
densenet=models.densenet121(pretrained=True)
input_size=densenet.classifier.in_features
out_put=10
densenet.classifier=nn.Linear(input_size,out_put)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

densenet.to(device)
#训练
optimizer=torch.optim.Adam(params=densenet.parameters(),lr=CIFAR10_Training_Parameters['lr'])


test_loader = get_cifar10_test_loader(dir_name='./RawModels/CIFAR10/',
                                          batch_size=CIFAR10_Training_Parameters['batch_size'])

train_loader, valid_loader = get_cifar10_train_validate_loader(dir_name='./RawModels/CIFAR10/',
                                                               batch_size=CIFAR10_Training_Parameters['batch_size'],
                                                               valid_size=0.1, shuffle=True)

# print(densenet)
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

def testing_evaluation(model, test_loader, device):
    """

    :param model:
    :param test_loader:
    :param device:
    :return:
    """
    print('\n#####################################')
    # print('#### The {} model is evaluated on the testing dataset loader ...... '.format(model.model_name))
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
    print('#### Accuracy of the loaded model on the testing dataset: {:.1f}/{:.1f} = {:.2f}%'.format(correct,
                                                                                                     total,
                                                                                                     ratio * 100))
    print('#####################################\n')

    return ratio


if __name__ =='__main__':
    for i in range(20):
        train_one_epoch(densenet,train_loader,optimizer,i,device)
        testing_evaluation(densenet,test_loader,device)

