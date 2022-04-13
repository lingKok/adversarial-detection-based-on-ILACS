from RawModels.ResNet import CIFAR10_Training_Parameters, adjust_learning_rate, resnet20_cifar,Net2
from RawModels.Utils.TrainTest import train_one_epoch, validation_evaluation, testing_evaluation, \
    train_one_epoch_by_NCE_cifar
from RawModels.Utils.dataset import get_cifar10_train_validate_loader, get_cifar10_test_loader
import torch
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
def get_advloader(dataset_type,attack_type):
    data_path='./AdversarialExampleDatasets/'+attack_type+'/'+dataset_type+'/'+attack_type+'_AdvExamples.npy'
    label_path='./AdversarialExampleDatasets/'+attack_type+'/'+dataset_type+'/'+attack_type+'_TrueLabels.npy'
    data=np.load(data_path)
    label=np.load(label_path)
    label=np.argmax(label,1)
    print(data.shape,label)
    data=torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    label=torch.from_numpy(label)
    adv_dataset=TensorDataset(data,label)
    adv_dataloader=DataLoader(adv_dataset)
    return adv_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# resnet_model=resnet20_cifar()
resnet_model=resnet20_cifar().to(device)
model_saver = './RawModels/CIFAR10/model/CIFAR10_' + 'raw' + '.pt'

    # resnet_model.load(path=model_saver,device=device)
final_model = copy.deepcopy(resnet_model)
final_model.load(path=model_saver, device=device)
dataset_type='CIFAR10'
attack_type='CW2'
adv_loader=get_advloader(dataset_type,attack_type)
test_loader=get_cifar10_test_loader(dir_name='./RawModels/CIFAR10/', batch_size=CIFAR10_Training_Parameters['batch_size'])
accuracy = testing_evaluation(model=final_model, test_loader=test_loader, device=device)
print('Finally, the ACCURACY of saved model [{}] on testing dataset is {:.2f}%\n'.format(final_model.model_name, accuracy * 100.0))
# for i,(data,label) in enumerate(test_loader):
#     for j in range(len(data)):
#         if j<10:
#             plt.imshow(data[j].permute(1,2,0))
#             plt.show()
#         break


