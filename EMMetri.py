import numpy as np
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as LPIPS
from torchmetrics.functional.image.rmse_sw import root_mean_squared_error_using_sliding_window as RMSE
from torchmetrics.functional.image.uqi import universal_image_quality_index as UIQI
import torch
import matplotlib.pyplot as plt

from tqdm import *

# def normalize_max_min(img):
#     Max = np.max(img)
#     Min = np.min(img)
#     return (img - Min) / ((Max - Min) + 1e-6)  #归一化

def normalize_max_min(img,gt):
    Max = np.max(gt)
    Min = np.min(gt)
    return (img - Min) / ((Max - Min) + 1e-6)  #归一化

def normalize(I):
    if isinstance(I,np.ndarray):
        normI = (I-np.amin(I,axis=(-1,-2)))/(np.amax(I,axis=(-1,-2))-np.amin(I,axis=(-1,-2)))
    elif isinstance(I,torch.Tensor):
        normI = (I-torch.amin(I,dim=(-1,-2)))/(torch.amax(I,dim=(-1,-2))-torch.amin(I,dim=(-1,-2)))
    else:
        raise TypeError("I must be cupy.ndarray or numpy.ndarray or torch.Tensor")
    return normI

# data = np.load('dataSetK3-CT-MLEM-192-Test.npz')
# data = np.load('./dataSetK3-CT-KEM-192-Ga-Test.npz')
# data = np.load('./dataSetBN-MR-Test.npz')
# data = np.load('./dataSetBN-OSEM-Test.npz')
data = np.load('./dataSetK3-CT-OSEM-192-Test.npz')
# data = np.load('./dataSetBN-RKEM-Test.npz')
input = data['input'][:,2]
# input  = np.load('datasetK3-PETCT-RKEM-192T-Ga-Test.npz')['kem'][:,0]
data = np.load('./ctpetraw.npz')
n_examples = len(data['PET'])
n_train = int(n_examples * 0.8) 

pet_gt = data['PET'][n_train:]
target = pet_gt

# input = np.clip(data['input'][:,2],0,5) / 5.0
# input = normalize_max_min(data['input'][:,1])
# input = data['input'][:,1]
# target = np.clip(data['target'],0,1000) / 1000
# target = data['target'][:,0:1]

print(np.max(input),np.min(input))
print(np.max(target),np.min(target))
total_psnr = 0
total_ssim = 0
total_lpip = 0
total_rmse = 0
total_uiqi = 0
qbar = trange(len(input))
for i in range(len(input)):
    input[i] = normalize_max_min(input[i]*30000,pet_gt[i])
    target[i] = normalize(target[i])
    # plt.subplot(1,2,1)
    # plt.imshow(input[i])
    # plt.subplot(1,2,2)
    # plt.imshow(target[i][0])
    # plt.show()
    inputi = input[i]
    # inputi = normalize_max_min(input[i])
    targeti = torch.Tensor(target[i]).unsqueeze(0).unsqueeze(0) 
    inputi = torch.Tensor(inputi).unsqueeze(0).unsqueeze(0) 


    psnr1 = PSNR(inputi,targeti,data_range=np.max(target[i]))
    ssim1 = SSIM(inputi,targeti,data_range=np.max(target[i]))
    rmse1 = RMSE(inputi,targeti)
    # lpip1 = LPIPS(predi.repeat(1,3,1,1),targeti.repeat(1,3,1,1),normalize=True)
    uiqi1 = UIQI(torch.ones_like(inputi)*0.00000001+inputi,targeti)
    # print(uiqi1)
    # if uiqi1 == torch.nan:
    #     uiqi1 = 0

    if torch.isnan(uiqi1):
        uiqi1 = 0
        plt.imshow(targeti[0,0])
        plt.show()
    total_psnr += psnr1
    total_ssim += ssim1
    # total_lpip += lpip1
    total_rmse += rmse1
    total_uiqi += uiqi1
    qbar.update(1)
    qbar.set_postfix(psnr=psnr1,ssim=ssim1) 



qbar.close()
total_psnr = total_psnr / len(input)
total_ssim = total_ssim / len(input)
total_lpip = total_lpip / len(input)
total_rmse = total_rmse / len(input)
total_uiqi = total_uiqi / len(input)

print(total_psnr.numpy(),total_ssim.numpy(),total_lpip,total_rmse.numpy(),total_uiqi.numpy())