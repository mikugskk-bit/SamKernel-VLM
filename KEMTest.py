import numpy as np
import matplotlib.pyplot as plt

from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as LPIPS
from torchmetrics.functional.image.rmse_sw import root_mean_squared_error_using_sliding_window as RMSE
from torchmetrics.functional.image.uqi import universal_image_quality_index as UIQI

import torch

from tqdm import *

def compare_mae(img1,img2,data_range=None):
    return np.mean(np.abs(img1-img2))


data = np.load("./datasetK3-PETCT-KEM-192T-Ga.npz")['kem']
data2 = np.load("./datasetK3-PETCT-KEM-192T.npz")['kem']
path = './ctpetraw.npz'

rdata = np.load(path)['PET']
rdataC = np.load(path)['CT']


n_examples = len(data)

n_train = int(n_examples * 0.8)  

testCases = data[n_train:]
testCases2 = data2[n_train:]
gt_pet = rdata[n_train:]
gt_ct = rdataC[n_train:]
print(data.shape)
# for i in range(1,int(n_examples * 0.2),1 ):
#     plt.subplot(1,4,1)
#     plt.title("KEM-CT Kernel")
#     plt.imshow(testCases[i,-1])
#     plt.subplot(1,4,2)
#     plt.title("KEM I Kernel")
#     plt.imshow(testCases2[i,-1])
#     plt.subplot(1,4,3)
#     plt.title("PET GT")
#     plt.imshow(gt_pet[i])
#     plt.subplot(1,4,4)
#     plt.title("CT GT")
#     plt.imshow(gt_ct[i])
#     plt.show()

# plt.subplot(1,4,1)
# plt.title("KEM-CT Kernel")
# plt.imshow(testCases[:200,-1,80,:])
# # plt.subplot(1,4,2)
# # plt.title("KEM I Kernel")
# # plt.imshow(testCases2[:200,-1,80,:])
# plt.subplot(1,4,2)
# plt.title("PET GT")
# plt.imshow(gt_pet[:200,80,:],vmax=30000)
# plt.subplot(1,4,3)
# plt.title("CT GT")
# plt.imshow(gt_ct[:200,80,:])
# plt.show()
pred = testCases[:,0]
target = np.clip(rdata/30000,0,1)
# input = data['input'][:,0]
# pred = data['pred'][:,0,16:176,16:176]
# target = data['target'][:,0,16:176,16:176]
# input = data['input'][:,0,16:176,16:176]

# print(input.shape)
print(pred.shape)
print(target.shape)

total_psnr = 0
total_ssim = 0
total_mae = 0
total_rmse = 0
total_uiqi = 0

c=0

qbar = trange(len(pred))
for i in range(len(pred)):


    # if np.sum(target[i]>1)>20:
    #     continue

    # if np.sum(target[i])<100:
    #     continue
    predi = torch.Tensor(pred[i]).unsqueeze(0).unsqueeze(0)
    targeti = torch.Tensor(target[i]).unsqueeze(0).unsqueeze(0)
    # inputi = torch.Tensor(input[i]).unsqueeze(0).unsqueeze(0)

    psnr1 = PSNR(torch.clip(predi,0,torch.max(targeti)),targeti,data_range=torch.max(targeti))
    # psnr1 = compare_psnr(target[i],pred[i].clip(0,np.max(target[i])),data_range=np.max(target[i]))
    mae = compare_mae(target[i],pred[i].clip(0,np.max(target[i])),data_range=np.max(target[i]))
    ssim1 = SSIM(torch.clip(predi,0,torch.max(targeti)),targeti,data_range=torch.max(targeti),kernel_size=11,gaussian_kernel=False)
    rmse1 = RMSE(torch.clip(predi,0,torch.max(targeti)),targeti)
    # lpip1 = LPIPS(predi.repeat(1,3,1,1),targeti.repeat(1,3,1,1),normalize=True)
    uiqi1 = UIQI(torch.clip(predi,0,torch.max(targeti)),targeti)
    total_psnr += psnr1
    total_ssim += ssim1
    total_mae += mae
    total_rmse += rmse1
    total_uiqi += uiqi1
    qbar.update(1)
    c=c+1



qbar.close()
total_psnr = total_psnr / c
total_ssim = total_ssim / c
total_lpip = total_mae / c
total_rmse = total_rmse / c
total_uiqi = total_uiqi / c

print(total_psnr.item(),total_ssim.item(),total_mae,total_rmse.item(),total_uiqi.item())
