import numpy as np
import matplotlib.pyplot as plt
attack_types=['FGSM','PGD','JSMA','DEEPFOOL','CW2']
path='./Feature/'+attack_types[4]+'.npy'
test_score,adv_score=np.load(path,allow_pickle=True)
print(adv_score)
rand_idx=np.random.choice(1000,100)
x=np.arange(0,100,1)
plt.plot(x,test_score[rand_idx],color = 'r',
         linestyle = '-.',
         linewidth = 1,
         marker = 'p',
         markersize = 5,
         markeredgecolor = 'b',
         markerfacecolor = 'r',
         label='Normal Examples')
plt.plot(x,adv_score[rand_idx],color = 'g',
         linestyle = '-.',
         linewidth = 1,
         marker = '*',
         markersize = 5,
         markeredgecolor = 'g',
         markerfacecolor = 'g',
         label='Adversarial Examples')
plt.xlabel('100 random CIFAR10 examples',fontsize=15)
plt.ylabel('IACS score',fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
# plt.legend(loc="lower right",fontsize=15,markerscale=2.0)
# plt.title(attack_types[4]+' on Resnet')
plt.show()