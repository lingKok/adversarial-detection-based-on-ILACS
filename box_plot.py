import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()      # 子图


# 封装一下这个函数，用来后面生成数据

#np.save('./Feature/' + attack_type + '.npy', [test_score, adv_score])
# 我们生成四组数据用来做实验，数据量分别为70-100
# 分别代表男生、女生在20岁和30岁的花费分布
attack_type=['FGSM','PGD','DEEPFOOL','JSMA','CW2']
test_fgsm,adv_fgsm=np.load('./Feature/' + attack_type[0] + '.npy').reshape(2,1000)
test_pgd,adv_pgd=np.load('./Feature/' + attack_type[1] + '.npy').reshape(2,1000)
test_deepfool,adv_deepfool=np.load('./Feature/' + attack_type[2] + '.npy').reshape(2,1000)
test_jsma,adv_jsma=np.load('./Feature/' + attack_type[3] + '.npy').reshape(2,1000)
test_cw2,adv_cw2=np.load('./Feature/' + attack_type[4] + '.npy').reshape(2,1000)

data = [test_fgsm, test_pgd, test_deepfool, test_jsma,test_cw2]
# data=[fgsm_test,fgsm_adv]
ax.boxplot(data,showfliers=False,patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
            # showmeans=True, # 以点的形式显示均值
            boxprops = {'color':'black','facecolor':'green'}, # 设置箱体属性，填充色和边框色
            # flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
            # meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
            # medianprops = {'linestyle':'--','color':'orange'}
)
value=[1,2,3,4,5]
data=[adv_fgsm,adv_pgd,adv_deepfool,adv_jsma,adv_cw2]
ax.boxplot(data,showfliers=False,patch_artist=True,boxprops={'color':'black','facecolor':'black'})
# plt.set_xticklabels(attack_type)
# plt.ylim(0,1)# 设置x轴刻度标签
plt.xticks(value,attack_type,fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('IACS score',fontsize=15)
plt.xlabel('Attack type',fontsize=15)
plt.show()
print(adv_deepfool)
# print(fgsm_adv.shape)