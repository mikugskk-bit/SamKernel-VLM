import torch
from MydataLoader import MyDatasetHRawSinoBrain,MyDatasetRawSinoBrain
from torch.utils.data import DataLoader
import numpy as np
from torch_radon import ParallelBeam

# train_dataset = MyDatasetHRawSinoBrain(is_train=True)
# test_dataset = MyDatasetHRawSinoBrain(is_train=False)
train_dataset = MyDatasetRawSinoBrain(is_train=True)
test_dataset = MyDatasetRawSinoBrain(is_train=False)


train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)

num_subsets = 1
total_iter = 30

det_count = 128
all_angles = np.linspace(0,np.pi,128,False)
subset_radons = []
sub_len = int(len(all_angles) / num_subsets)

for i in range(num_subsets): 
    angles = all_angles[i*sub_len:(i+1)*sub_len]
    sub_radon = ParallelBeam(det_count,angles)
    subset_radons.append(sub_radon)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
preds = []
for input,target in train_dataloader:

    inp = input.to(device)

    sinogram = inp[:,0]
    mul_factor = inp[:,1]
    pet_noise = inp[:,2]

    num_subsets = 1
    total_iter = 30
    pet_gt1 = torch.zeros((inp.shape[0],1,128,128)).cuda()
    sinogram = sinogram.reshape(pet_gt1.shape)
    mul_factor = mul_factor.reshape(pet_gt1.shape)
    pet_noise = pet_noise.reshape(pet_gt1.shape)

    subset_radons = subset_radons

    # 迭代重建
    sub_pet_eval = torch.ones_like(pet_gt1).reshape(inp.shape[0],1,-1,1)

    mask_list = []
    points_list = []
    boxs_list = []

    ic = 0
    for i in range(total_iter):
        for subset in range(num_subsets):
            sub_sinogram = sinogram[:,:,subset*sub_len:(subset+1)*sub_len].reshape(inp.shape[0],1,-1,1)
            sub_mul_factor = mul_factor[:,:,subset*sub_len:(subset+1)*sub_len]
            sub_pet_noise = pet_noise[:,:,subset*sub_len:(subset+1)*sub_len]
            G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

            l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape))
            sinogram_eval = (sub_mul_factor.reshape(inp.shape[0],1,-1,1) * l.reshape(inp.shape[0],1,-1,1) + sub_pet_noise.reshape(inp.shape[0],1,-1,1))
            tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
            # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
            backproj = subset_radons[subset].backward((sub_mul_factor.reshape(inp.shape[0],1,-1,1)*tmp).reshape(sub_mul_factor.shape))
            sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(inp.shape[0],1,-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(inp.shape[0],1,-1,1)

    for i in range(len(input)):
        preds.append(sub_pet_eval[i].detach().cpu())


np.savez('./newDataset/osem-50e4-30-1-train.npz',osem=preds)


preds = []
for input,target in test_dataloader:

    inp = input.to(device)

    sinogram = inp[:,0]
    mul_factor = inp[:,1]
    pet_noise = inp[:,2]

    num_subsets = 1
    total_iter = 30
    pet_gt1 = torch.zeros((inp.shape[0],1,128,128)).cuda()
    sinogram = sinogram.reshape(pet_gt1.shape)
    mul_factor = mul_factor.reshape(pet_gt1.shape)
    pet_noise = pet_noise.reshape(pet_gt1.shape)

    subset_radons = subset_radons

    # 迭代重建
    sub_pet_eval = torch.ones_like(pet_gt1).reshape(inp.shape[0],1,-1,1)

    mask_list = []
    points_list = []
    boxs_list = []

    ic = 0
    for i in range(total_iter):
        for subset in range(num_subsets):
            sub_sinogram = sinogram[:,:,subset*sub_len:(subset+1)*sub_len].reshape(inp.shape[0],1,-1,1)
            sub_mul_factor = mul_factor[:,:,subset*sub_len:(subset+1)*sub_len]
            sub_pet_noise = pet_noise[:,:,subset*sub_len:(subset+1)*sub_len]
            G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

            l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape))
            sinogram_eval = (sub_mul_factor.reshape(inp.shape[0],1,-1,1) * l.reshape(inp.shape[0],1,-1,1) + sub_pet_noise.reshape(inp.shape[0],1,-1,1))
            tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
            # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
            backproj = subset_radons[subset].backward((sub_mul_factor.reshape(inp.shape[0],1,-1,1)*tmp).reshape(sub_mul_factor.shape))
            sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(inp.shape[0],1,-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(inp.shape[0],1,-1,1)

    for i in range(len(input)):
        preds.append(sub_pet_eval[i].detach().cpu())


np.savez('./newDataset/osem-50e4-30-1-test.npz',osem=preds)