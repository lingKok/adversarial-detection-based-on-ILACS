# import numpy as np
#
# import matplotlib.pyplot as plt
# y=[0.6082,0.6023,0.6102,0.6014,0.5959,0.6165,0.7288,0.7776,0.8964,0.7153,0.9253,0.9324,0.9636,0.7245]#
# x=np.arange(0,14,1)
# plt.plot(x,y,color = 'r',
#          linestyle = '-.',
#          linewidth = 1,
#          marker = 'p',
#          markersize = 5,
#          markeredgecolor = 'b',
#          markerfacecolor = 'r',
#          label='normal examples')
# _x_ticks = ["layer{}".format(i+1) for i in x ]
#
# plt.xticks(x[::1], _x_ticks[::1], rotation=45)
# plt.xlabel('Layers for ILACS estimation')
# plt.ylabel('ROC score')
# plt.ylim((0,1))
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# x = np.random.rand(100).reshape(10,10)
attacks=['FGSM','PGD','DeepFool','JSMA','CW2']
y=np.asarray([])
x=np.asarray([[0.987	,0.9871	,0.9837	,0.9846	,0.983]
             ,[0.9651	,0.9914	,0.9828	,0.9856	,0.9828],
[0.9712	,0.9885	,0.986	,0.9855	,0.9832],
              [0.8083	,0.8069	,0.9108	,0.9884	,0.9847],
              [0.8275	,0.8177	,0.9356	,0.9887	,0.9857],

              ])
y=np.asarray([[0.9947	,0.615	,0.559	,0.5821	,0.5793]
             ,[0.5103	,0.9955	,0.4961	,0.5088	,0.5088],
              [0.9815	,0.9637	,0.9108	,0.6667	,0.8534],
              [0.9435	,0.9432	,0.6832	,0.7571	,0.6718],
              [0.9749	,0.9249	,0.8355	,0.6373	,0.928]])
z=y=np.asarray([[0.7277	,0.5708	,0.5215	,0.4847	,0.5262]
             ,[0.5412	,0.9769	,0.5243	,0.4916	,0.5303],
              [0.5356	,0.5708	,0.6615	,0.4871	,0.5259],
              [0.5384	,0.5708	,0.5215	,0.5612	,0.5262],
              [0.5385	,0.5708	,0.523	,0.4909	,0.7005]])
fig=plt.figure(figsize=(20,6))
c1=plt.subplot(131)
plt.imshow(x,cmap=plt.cm.summer,vmin=0,vmax=1)
plt.xticks([i for i in range(len(attacks))],attacks,fontsize=15)
plt.yticks([i for i in range(len(attacks))],attacks,fontsize=15)
plt.title('IACS',fontsize=20)
plt.subplot(132)
c2=plt.imshow(y,cmap=plt.cm.summer,vmin=0,vmax=1)
plt.xticks([i for i in range(len(attacks))],attacks,fontsize=15)
# plt.yticks([i for i in range(len(attacks))],attacks,fontsize=18)
plt.yticks([])
plt.title('LID',fontsize=20)
plt.subplot(133)
c3=plt.imshow(z,cmap=plt.cm.summer,vmin=0,vmax=1)
plt.xticks([i for i in range(len(attacks))],attacks,fontsize=15)
# plt.yticks([i for i in range(len(attacks))],attacks,fontsize=18)
plt.yticks([])
plt.title('KD',fontsize=20)
fig.subplots_adjust(right=0.9)
#colorbar 左 下 宽 高
l = 0.92
b = 0.12
w = 0.015
h = 1 - 2*b

#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h]
cbar_ax = fig.add_axes(rect)
cb = plt.colorbar(c3, cax=cbar_ax)

#设置colorbar标签字体等
cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
font = {'family' : 'serif',
#       'color'  : 'darkred',
    'color'  : 'black',
    'weight' : 'normal',
    'size'   : 16,
    }
 #设置colorbar的标签字体及其大小
# plt.tight_layout()
plt.show()


