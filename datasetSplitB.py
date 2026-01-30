import numpy as np


train_id = np.load('./train_list1.npy')
test_id = np.load('./test_list1.npy')
print(train_id.shape,test_id.shape)
data = np.load('./dataSetB3N-RKEM.npz')
# data2 = np.load('F:\Codes\pet_vol_slice_144by144_bwpm.npy')
print(data['kem'].shape)
# np.random.seed(42)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
# n_examples = len(data['kem'])
# n_examples = 1000
# n_train = int(n_examples * 0.8)  
# train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
# test_idx = list(set(range(0,n_examples))-set(train_idx))

# 5.1 训练还是测试数据集

input = data['kem'][train_id,:]
target = data['pet'][train_id]
print(input.shape)
np.savez('./dataSetBN-RKEM-Train.npz',input=input,target = target)


input = data['kem'][test_id,:]
target = data['pet'][test_id]
np.savez('./dataSetBN-RKEM-Test.npz',input=input,target = target)