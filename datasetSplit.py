import numpy as np


data = np.load('./dataSetK3-T-KEM-192.npz')
np.random.seed(42)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
n_examples = len(data['pet'])
# n_examples = 1000
n_train = int(n_examples * 0.8)  
# train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
# test_idx = list(set(range(0,n_examples))-set(train_idx))

# 5.1 训练还是测试数据集

input = data['kem'][:n_train]
target = data['pet'][:n_train]
np.savez('./dataSetK3-T-KEM-192-Train.npz',input=input,target = target)


input = data['kem'][n_train:]
target = data['pet'][n_train:]
np.savez('./dataSetK3-T-KEM-192-Test.npz',input=input,target = target)