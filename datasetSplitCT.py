import numpy as np
from tqdm import *


def normalize_max_min(img):
    Max = np.max(img)
    Min = np.min(img)
    return (img - Min) / ((Max - Min) + 1e-6)  #归一化


dataPETCT = np.load('./ctpetraw.npz')
KEM = np.load('./datasetK3-PETCT-OSEM-192T.npz')



print(dataPETCT, KEM)

pet = dataPETCT['PET']
ct = dataPETCT['CT']

kem = KEM['kem']


qbar = trange(0,len(pet))
for i in range(len(pet)):
    # pet[i] = normalize_max_min(pet[i])
    pet[i] = np.clip(pet[i]/30000,0,1)
    ct[i] = normalize_max_min(ct[i])
    # for j in range(3):
    #     kem[i,j] = normalize_max_min(kem[i,j])
    qbar.update(1)
qbar.close()

print(pet.shape,ct.shape,kem.shape)

# input = np.concatenate((np.expand_dims(ct,axis=1),kem),axis=1)

input = kem
target = np.expand_dims(pet,axis=1)

print(input.shape,target.shape)

n_examples = len(input)

n_train = int(n_examples * 0.8)  


# 5.1 训练还是测试数据集

input1 = input[:n_train]
target1 = target[:n_train]
print(input1.shape,target1.shape)
np.savez('./dataSetK3-CT-OSEM-192-Train.npz',input=input1,target = target1)


input2 = input[n_train:]
target2 = target[n_train:]
print(input2.shape,target2.shape)
np.savez('./dataSetK3-CT-OSEM-192-Test.npz',input=input2,target = target2)