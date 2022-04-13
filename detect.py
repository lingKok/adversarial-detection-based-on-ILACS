import torch
import os
import argparse
import numpy as np
import random




from RawModels.Utils.TrainTest import testing_evaluation, predict
from RawModels.Utils.dataset import get_mnist_test_loader, get_mnist_train_validate_loader,get_svhn_train_validate_loader,get_svhn_test_loader\
    ,get_cifar10_train_validate_loader,get_cifar10_test_loader
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


def plot_auc(y_true, y_pre,attack_type,dataset_type,detect_type):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, threshold = roc_curve(y_true, y_pre)
    roc_auc = auc(fpr, tpr)
    path='./ROC/'+dataset_type+'/'+detect_type+'_'+attack_type+'.npy'
    np.save(path,[fpr,tpr,roc_auc])
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot(point[0], point[1], marker='o', color='r')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    # print('a is :', a.shape)
    return a


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积

    res = num / (denom+1e-10)
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res
def get_distance_matrix(v1,v2):
    size=len(v1)
    assert len(v1)==len(v2)
    v1=v1.reshape(size,1,-1)
    v2=v2.reshape(1,size, -1)
    v1_min_v2=v1-v2
    return np.sqrt(np.einsum('ijx,ijx->ij',v1_min_v2,v1_min_v2))


def cos_sim(data, batch, k):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: np.mean(v)
    a = get_cos_similar_matrix(data, batch)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, -k - 2:-2]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a
def knn_sim(data, batch, k):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: np.mean(v)
    a = get_distance_matrix(data, batch)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a
def cos_sim_global(data, batch, k):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: np.mean(v)
    a = get_cos_similar_matrix(data, batch)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a
def cos_sim_in_class(data, label, batch, batch_label):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    assert len(data) == len(batch)
    n = len(data)
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    # k = min(k, len(data) - 1)
    a = get_cos_similar_matrix(batch, data)
    a = torch.from_numpy(a)
    # print(a.shape)
    # print('label is :', label)
    # print('adv_label is :', batch_label)
    mask = torch.ones_like(a) * (batch_label.expand(n, n).eq(label.expand(n, n).t()))
    duijiao_0 = torch.ones_like(mask) - torch.eye(n)  # 为了代码方便可能丢失对角线上label相同的值，但是对结果不会造成影响。
    mask = mask * duijiao_0
    # print(torch.sum(mask, 1))
    a = torch.sort(a * mask, 1, descending=True)[0][:, 0:10]
    a = torch.mean(a, 1)
    return a.numpy()


def cos_sim_in_class_global(data, label, batch, batch_label):
    """
    input: shape is (n,m) n is batch_size,m is dimension
    output:shape is (n,n)
    """
    assert len(data) == len(batch)
    n = len(data)
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    # k = min(k, len(data) - 1)
    a = get_cos_similar_matrix(batch, data)
    a = torch.from_numpy(a)
    # print(a.shape)
    # print('label is :', label)
    # print('adv_label is :', batch_label)
    mask = torch.ones_like(a) * (batch_label.expand(n, n).eq(label.expand(n, n).t()))
    duijiao_0 = torch.ones_like(mask) - torch.eye(n)  # 为了代码方便可能丢失对角线上label相同的值，但是对结果不会造成影响。
    mask = mask * duijiao_0
    # print(torch.sum(mask, 1))
    a = torch.sort(a * mask, 1, descending=True)[0]
    a = torch.sum(a, 1)/torch.sum(mask,1)
    return a.numpy()


def get_lid_by_random(data, adv_data, batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]

        X_adv_act = adv_data[start:end]
        lid_batch = mle_batch(X_act, X_act, k)
        lid_batch_adv = mle_batch(X_act, X_adv_act, k)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv


def get_ilacs_by_random(data, label1, adv_data, label2, batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim_in_class(X_act - X_mean, label1, X_act - X_mean, label1)
        lid_batch_adv = cos_sim_in_class(X_act - X_mean, label1, X_adv_act - X_mean, label2)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv

def get_ilcs_by_random(data, label1, adv_data, label2, batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        # X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim_in_class(X_act , label1, X_act , label1)
        lid_batch_adv = cos_sim_in_class(X_act , label1, X_adv_act , label2)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv
def get_iacs_by_random(data, label1, adv_data, label2, batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim_in_class_global(X_act - X_mean, label1, X_act - X_mean, label1)
        lid_batch_adv = cos_sim_in_class_global(X_act - X_mean, label1, X_adv_act - X_mean, label2)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv

def get_ics_by_random(data, label1, adv_data, label2, batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        # X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim_in_class_global(X_act , label1, X_act , label1)
        lid_batch_adv = cos_sim_in_class_global(X_act , label1, X_adv_act , label2)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv

def get_lacs_by_random(data,  adv_data,  batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim(X_act - X_mean,  X_act - X_mean,k)
        lid_batch_adv = cos_sim(X_act - X_mean,  X_adv_act - X_mean,k)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv

def get_acs_by_random(data,  adv_data,  batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim_global(X_act - X_mean,  X_act - X_mean,k)
        lid_batch_adv = cos_sim_global(X_act - X_mean,  X_adv_act - X_mean,k)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv

def get_cs_by_random(data,  adv_data,  batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        # X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim_global(X_act,  X_act ,k)
        lid_batch_adv = cos_sim_global(X_act ,  X_adv_act ,k)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv
def get_lcs_by_random(data,  adv_data,  batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        # X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = cos_sim(X_act,  X_act ,k)
        lid_batch_adv = cos_sim(X_act ,  X_adv_act ,k)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv

def get_knn_by_random(data,  adv_data,  batch_size, k=10):
    batch_num = int(np.ceil(data.shape[0] / float(batch_size)))
    lid = None
    lid_adv = None
    for i_batch in range(batch_num):
        start = i_batch * batch_size
        end = np.minimum(len(data), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, 1))
        lid_batch_adv = np.zeros(shape=(n_feed, 1))
        X_act = data[start:end]
        # X_mean = np.mean(X_act, 0)
        X_adv_act = adv_data[start:end]
        lid_batch = knn_sim(X_act,  X_act ,k)
        lid_batch_adv = knn_sim(X_act ,  X_adv_act ,k)
        if lid is not None:
            lid = np.concatenate((lid, lid_batch), 0)
            lid_adv = np.concatenate((lid_adv, lid_batch_adv), 0)
        else:
            lid = lid_batch
            lid_adv = lid_batch_adv
    return lid, lid_adv
def get_score_by_ILACS(dict1, label1, dict2, label2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_ilacs_by_random(feature1, label1, feature2, label2, batch_size=1000, k=10)
    return base_score, adv_score

def get_score_by_ILCS(dict1, label1, dict2, label2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_ilcs_by_random(feature1, label1, feature2, label2, batch_size=1000, k=10)
    return base_score, adv_score

def get_score_by_IACS(dict1, label1, dict2, label2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_iacs_by_random(feature1, label1, feature2, label2, batch_size=1000, k=10)
    return base_score, adv_score

def get_score_by_ICS(dict1, label1, dict2, label2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_ics_by_random(feature1, label1, feature2, label2, batch_size=1000, k=10)
    return base_score, adv_score
def get_score_by_LACS(dict1,  dict2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_lacs_by_random(feature1,  feature2,  batch_size=100, k=10)
    return base_score, adv_score

def get_score_by_ACS(dict1,  dict2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_acs_by_random(feature1,  feature2,  batch_size=100, k=10)

    return base_score, adv_score


def get_score_by_CS(dict1, dict2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_cs_by_random(feature1, feature2, batch_size=100, k=10)

    return base_score, adv_score
def get_score_by_LCS(dict1, dict2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_lcs_by_random(feature1, feature2, batch_size=100, k=10)

    return base_score, adv_score
def get_score_by_KNN(dict1, dict2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_knn_by_random(feature1, feature2, batch_size=100, k=10)

    return base_score, adv_score
def get_score_by_lid(dict1, dict2):
    assert len(dict1) == len(dict2)
    print(dict1[0][0])
    print('####################################')
    print(dict2[0][0])
    print('####################################')
    dim = len(dict1)
    base_score = np.zeros((len(dict1[0]), dim))
    adv_score = np.zeros((len(dict1[0]), dim))
    for i in range(dim):
        feature1 = dict1[i].detach().numpy()

        print('feature1 is :', feature1[0])
        feature2 = dict2[i].detach().numpy()
        print('feature2 is :', feature2[0])

        base_score[:, i], adv_score[:, i] = get_lid_by_random(feature1, feature2, batch_size=100, k=10)
    return base_score, adv_score


def score_point(dup):
    x, kde = dup
    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs):
    import multiprocessing as mp
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    result = np.asarray(
        p.map(score_point, [(x, kdes[preds]) for x, preds in zip(samples, preds)])
    )
    p.close()
    p.join()
    return result


# def get_test_feature(model, data_loader, device):
#     None_feature()
#     testing_evaluation(model, data_loader, device)
#     feature = get_feature()
#     return feature


def adv_loader(type, dataset, batch_size):
    from torch.utils.data import TensorDataset, DataLoader
    advdata_path = './AdversarialExampleDatasets/' + type + '/' + dataset + '/' + type + '_AdvExamples.npy'
    Advlabel_path = './AdversarialExampleDatasets/' + type + '/' + dataset + '/' + type + '_AdvLabels.npy'
    adv_data = np.load(advdata_path)
    adv_label = np.load(Advlabel_path)
    adv_data = torch.from_numpy(adv_data)
    adv_label = torch.from_numpy(adv_label)
    adv_data = adv_data.to(torch.float32)
    # _, true_label = torch.max(true_label, 1)
    # print(true_label.shape)
    adv_dataset = TensorDataset(adv_data, adv_label)
    adv_loader = DataLoader(adv_dataset, batch_size=batch_size)
    return adv_loader


def adv_label(attack_type, dataset_type):
    path = './AdversarialExampleDatasets/' + attack_type + '/' + dataset_type + '/' + attack_type + '_AdvLabels.npy'
    adv_labels = np.load(path)
    adv_labels = torch.from_numpy(adv_labels)
    return adv_labels


def get_trainlabel(train_dataloader):
    label_arr = None
    for data, label in train_dataloader:
        if label_arr is not None:
            label_arr = np.concatenate((label_arr, label), 0)
        else:
            label_arr = label
    return label_arr


def mix_feature(test_score, adv_score):
    adv_label = np.zeros(len(adv_score))
    test_label = np.ones(len(test_score))
    mix_data = np.concatenate((test_score, adv_score), 0)
    mix_label = np.concatenate((test_label, adv_label), 0)
    return mix_data, mix_label


def get_data(dataloader, size):
    data_arr = None
    label_arr = None
    for data, label in dataloader:
        if data_arr is not None:
            data_arr = torch.cat((data_arr, data), 0)
            label_arr = torch.cat((label_arr, label), 0)
        else:
            data_arr = data
            label_arr = label

    return data_arr[0:size], label_arr[0:size]


def get_kde_score(train_feature, train_labels, test_feature, test_labels, adv_feature, adv_labels):
    adv_labels = adv_labels.numpy()
    test_labels = test_labels.numpy()
    class_idx = {}
    for i in range(10):
        class_idx[i] = np.where(train_labels == i)
        print(class_idx[i][0].shape)
    kdes = {}

    for i in range(10):
        kdes[i] = KernelDensity(bandwidth=0.26, kernel='gaussian').fit(train_feature[class_idx[i]],
                                                                       train_labels[class_idx[i]])
    test_score = score_samples(kdes, test_feature, test_labels, n_jobs=1)
    test_score = test_score.reshape((-1, 1))
    adv_ind=np.isnan(adv_feature)
    adv_feature[adv_ind]=0
    adv_score = score_samples(kdes, adv_feature, adv_labels, n_jobs=1)
    adv_score = adv_score.reshape((-1, 1))

    return test_score, adv_score



def main():
    import argparse
    parse= argparse.ArgumentParser(description='this is an adversarial detection based on Local Cosine Similarity')
    parse.add_argument('--seed', type=int, default=100, help='set the random seed')
    parse.add_argument('--gpu_index', type=str, default='1', help='the index of gpu')
    parse.add_argument('--dataset',default='mnist',choices=['MNIST','SVHN','CIFAR10'])
    parse.add_argument('--type',default='cs',choices=['knn','lcs','cs','kd','lid'])
    parse.add_argument('--attack',default='fgsm',choices=['FGSM','PGD','DEEPFOOL','JSMA','CW2'])
    parse.add_argument('--test_attack', default='fgsm', choices=['FGSM', 'PGD', 'DEEPFOOL', 'JSMA', 'CW2'])
    args=parse.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset=='MNIST':
        from RawModels.MNISTConv import MNISTConvNet, MNIST_Training_Parameters, None_feature, get_feature
        rawModel_path = 'RawModels/MNIST/model/MNIST_raw.pt'

        test_loader = get_mnist_test_loader(dir_name='./RawModels/MNIST/',
                                        batch_size=MNIST_Training_Parameters['batch_size'])
        train_loader, _ = get_mnist_train_validate_loader(dir_name='./RawModels/MNIST/',
                                                      batch_size=MNIST_Training_Parameters['batch_size'])
        raw_model = MNISTConvNet().to(device)
    if args.dataset == 'SVHN':
        from RawModels.SVHNConv import SVHNConvNet, SVHN_Training_Parameters, None_feature, get_feature
        rawModel_path = 'RawModels/SVHN/model/SVHN_raw.pt'
        test_loader = get_svhn_test_loader(dir_name='./RawModels/SVHN/',
                                           batch_size=SVHN_Training_Parameters['batch_size'])
        train_loader,_ = get_svhn_train_validate_loader(dir_name='./RawModels/SVHN/',
                                                        batch_size=SVHN_Training_Parameters['batch_size'])
        raw_model = SVHNConvNet().to(device)
    if args.dataset == 'CIFAR10':
        from RawModels.ResNet import resnet20_cifar, CIFAR10_Training_Parameters, None_feature, get_feature
        rawModel_path = 'RawModels/CIFAR10/model/CIFAR10_raw.pt'
        test_loader = get_cifar10_test_loader(dir_name='./RawModels/CIFAR10/',
                                              batch_size=CIFAR10_Training_Parameters['batch_size'])
        train_loader, _ =get_cifar10_train_validate_loader(dir_name='./RawModels/CIFAR10/',
                                                           batch_size=CIFAR10_Training_Parameters['batch_size'])
        raw_model=resnet20_cifar().to(device)
    if not os.path.exists(rawModel_path):
        print('please train model first!')

    raw_model.load(path=rawModel_path, device=device)
    attack_type = args.attack
    dataset_type = args.dataset
    detect_type= args.type

    advloader = adv_loader(attack_type, dataset_type, 100)
    None_feature()
    test_label = predict(raw_model, test_loader, device)[0:1000]
    test_kde_feature = get_feature()[0:1000]

    None_feature()
    adv_label = predict(raw_model, advloader,device)
    adv_kde_feature = get_feature()
    # print(test_kde_feature.shape)
    test_data, _ = get_data(test_loader, 1000)
    raw_model(test_data)
    test_feature_all = raw_model.middle.copy()
    test_feature_single = raw_model.mid_layer.copy()

    adv_data, _ = get_data(advloader, 1000)

    raw_model(adv_data)
    adv_feature_all = raw_model.middle.copy()
    adv_feature_single = raw_model.mid_layer.copy()
    if detect_type == 'lid':
        test_score, adv_score = get_score_by_lid(test_feature_all, adv_feature_all)
    elif detect_type == 'kd':

        print('nan:', torch.isnan(adv_kde_feature).any())
        print('inf:', torch.isinf(adv_kde_feature).any())
        None_feature()
        train_label = predict(raw_model, train_loader, device)
        train_kde_feature = get_feature()
        test_score, adv_score = get_kde_score(train_kde_feature, train_label, test_kde_feature, test_label,
                                              adv_kde_feature, adv_label)
    elif detect_type == 'ilacs':
        test_score, adv_score = get_score_by_ILACS(test_feature_all, test_label, adv_feature_all, adv_label)
        np.save('./Feature/' + attack_type + '.npy', [test_score, adv_score])
    elif detect_type == 'lacs':
        test_score, adv_score = get_score_by_LACS(test_feature_all, adv_feature_all)
    elif detect_type == 'acs':
        test_score, adv_score = get_score_by_ACS(test_feature_all, adv_feature_all)
    elif detect_type == 'cs':
        test_score, adv_score = get_score_by_CS(test_feature_all, adv_feature_all)
    elif detect_type == 'lcs':
        test_score, adv_score = get_score_by_LCS(test_feature_all, adv_feature_all)
    elif detect_type == 'knn':
        test_score, adv_score = get_score_by_KNN(test_feature_all, adv_feature_all)
    elif detect_type == 'iacs':
        test_score, adv_score = get_score_by_IACS(test_feature_all, test_label, adv_feature_all, adv_label)
        print('adv score :',adv_score)
    elif detect_type == 'ics':
        test_score, adv_score = get_score_by_ICS(test_feature_all, test_label, adv_feature_all, adv_label)
    elif detect_type=='ilcs':
        test_score, adv_score = get_score_by_ILCS(test_feature_single, test_label, adv_feature_single, adv_label)
    else:
        print('input right detect type')
        exit(0)
    print('test score is :', test_score[0:5])
    print(adv_score[0:5])
    mix_data, mix_label = mix_feature(test_score, adv_score)
    test_attack_type=args.test_attack
    if attack_type != test_attack_type:
        path = './Feature/' + detect_type + '_' + attack_type + '.npy'
        if os.path.exists(path):
            test_score, adv_score = np.load(path)
        else:
            print('first generate the feature of attack!')
            exit()

        mix_data, mix_label = mix_feature(test_score, adv_score)
        print(mix_data.shape)
        scale = MinMaxScaler().fit(mix_data)
        mix_data = scale.transform(mix_data)
        x_train, _, y_train, _ = train_test_split(mix_data, mix_label, random_state=0, test_size=0.33)
        path1 = './Feature/' + detect_type + '_' + test_attack_type + '.npy'
        test_score1, adv_score1 = np.load(path1)
        mix_data1, mix_label1 = mix_feature(test_score1, adv_score1)
        _, x_test, _, y_test = train_test_split(mix_data1, mix_label1, random_state=0, test_size=0.33)
    else:
        # path = './Feature/' + detect_type + '_' + attack_type + '.npy'
        # test_score, adv_score = np.load(path)
        mix_data, mix_label = mix_feature(test_score, adv_score)
        x_train, x_test, y_train, y_test = train_test_split(mix_data, mix_label, random_state=0, test_size=0.33)
    x_train_idx = np.isnan(x_train)
    x_test_idx = np.isnan(x_test)
    x_train[x_train_idx] = 0
    x_test[x_test_idx] = 0
    print('nan:', np.isnan(x_train).any())
    print('inf:', np.isinf(x_train).any())
    lr = LogisticRegressionCV(max_iter=1000).fit(x_train, y_train)
    predict_score = lr.predict_proba(x_test)[:, 1]
    plot_auc(y_test, predict_score, attack_type, dataset_type, detect_type)


if __name__ == '__main__':

    main()
