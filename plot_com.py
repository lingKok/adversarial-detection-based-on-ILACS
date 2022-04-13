from typing import ChainMap
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
# dataset_types=['MNIST','SVHN','CIFAR10']
dataset_types=['CIFAR10']
detect_types=['IACS','LID','KD']
# attack_types=['FGSM','PGD','JSMA','DEEPFOOL','CW2']
attack_types=['DEEPFOOL']
# path='./ROC/'+dataset_type+'/'+detect_type+'_'+attack_type+'.npy'

color = ['blue', 'red', 'green']
linestyle = ['-', '--', ':']


def plot(file1, file2, file3,type):

    plt.figure(figsize=(10, 10))

    plt.title(type,fontsize=20)
    fpr0, tpr0, roc_auc0 = np.load(file1, allow_pickle=True)
    fpr1, tpr1, roc_auc1 = np.load(file2,allow_pickle=True)
    fpr2, tpr2, roc_auc2 = np.load(file3, allow_pickle=True)
    plt.plot(fpr0,
             tpr0,
             color=color[0],
             lw=2,
             linestyle=linestyle[0],
             label='ROC of IACS curve(area = %0.4f)' %
             (roc_auc0))  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot(fpr1,
             tpr1,
             color=color[1],
             lw=2,
             linestyle=linestyle[0],
             label='ROC of LID curve (area = %0.4f)' %
             (roc_auc1))  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot(fpr2,
             tpr2,
             color=color[2],
             lw=2,
             linestyle=linestyle[0],
             label='ROC of KD curve (area = %0.4f)' %
                   (roc_auc2))  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot(point[0], point[1], marker='o', color='r')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR',fontsize=20)
    plt.ylabel('TPR',fontsize=20)
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right",fontsize=20,markerscale=2.0)
    path='./ROC'+type+'.png'
    plt.savefig(path)


for dataset_type in dataset_types:
    for attack_type in attack_types:
        i=0
        path={}
        for detect_type in detect_types:
            path[i]='./ROC/'+dataset_type+'/'+detect_type+'_'+attack_type+'.npy'
            print(i)
            print(path[i])
            i=i+1
        plot(path[0],path[1],path[2],dataset_type+'_'+attack_type)
