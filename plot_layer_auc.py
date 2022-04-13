import numpy as np

import matplotlib.pyplot as plt
y1=[0.7307,0.7407,0.7603,0.8036,0.8866,0.9331,0.9879,0.9926,0.9862,0.9833,0.9877,0.9896,0.9795,0.9289]#res_iacs
y2=[0.5739,0.5868,0.5893,0.6122,0.6170,0.8163,0.9295,0.9847,0.9868,0.9863,0.9880,0.9897,0.9615,0.9859]
#y=[0.7356,0.8785,0.9617,0.9734,0.9815,0.9768,0.9704,0.9704,0.9808,0.9854,0.9334]conv_iacs
# y=[0.5669,0.7251,0.9250,0.9335,0.9697,0.9572,0.9431,0.9431,0.9690,0.9826,0.9791]#conv_ics
x=np.arange(0,14,1)
plt.plot(x,y1,color = 'r',
         linestyle = '-.',
         linewidth = 1,
         marker = 'p',
         markersize = 5,
         markeredgecolor = 'r',
         markerfacecolor = 'r',
         label='Detector with IACS')
plt.plot(x,y2,color = 'b',
         linestyle = '-',
         linewidth = 1,
         marker = 'p',
         markersize = 5,
         markeredgecolor = 'b',
         markerfacecolor = 'b',
         label='Detector with ICS',
         )
_x_ticks = ["{}".format(i+1) for i in x ]

plt.xticks(x[::1], _x_ticks[::1],fontsize=12)
plt.xlabel('Layer',fontsize=15)
plt.ylabel('AUC score',fontsize=15)
plt.legend(loc="lower right",fontsize=15,markerscale=2.0)
plt.ylim((0.5,1))
plt.yticks(fontsize=12)
plt.show()