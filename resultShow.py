import numpy as np
import matplotlib.pyplot as plt

import torch

from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM

from tqdm import *


# def PSNR(pred, true, min_max_norm=True):
#     """Peak Signal-to-Noise Ratio.

#     Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#     """
#     mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
#     if mse == 0:
#         return float('inf')
#     else:
#         if min_max_norm:  # [0, 1] normalized by min and max
#             return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
#         else:
#             return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std

models = ['EM-U','EM-SAM','EM-SAMF','EM-SAM-A','EM-U-SAM','EM-U-SAM-G','EM-U-P','EM-U-TP','EM-U-SAM-G3CP-GANC']
cmodels = ['CNNBPNet','FBPNet_prior','ADMMNet','IterativeNet']

model_name = models[-1]
data = np.load('result/'+model_name+'.npz')



print(data['input'].shape)
pred = data['pred'][:,0]
target = data['target'][:,0]
input = data['input'][:,0]
# input = data['input'].reshape(target.shape[0],3,target.shape[1] * 2,target.shape[2] * 2)

# input = np.expand_dims(input,axis=3)
# input = input.repeat(axis=3,repeats=3)
print(input.shape)
print(pred.shape)
print(target.shape)

total_psnr = 0
total_ssim = 0
# for i in range(len(pred)):

#     psnr1 = PSNR(pred[i],target[i],data_range=1.0)
#     ssim1 = SSIM(pred[i],target[i],data_range=1.0)
#     total_psnr += psnr1
#     total_ssim += ssim1
# total_psnr = total_psnr / len(pred)
# total_ssim = total_ssim / len(pred)
# print(total_psnr,total_ssim)


for i in range(0,len(pred),10):
    predi = torch.Tensor(pred[i]).unsqueeze(0).unsqueeze(0)
    targeti = torch.Tensor(target[i]).unsqueeze(0).unsqueeze(0)
    inputi = torch.Tensor(input[i]).unsqueeze(0).unsqueeze(0)

    psnr1 = PSNR(predi,targeti,data_range=torch.max(targeti))
    ssim1 = SSIM(predi,targeti,data_range=torch.max(targeti))



    psnr2 = PSNR(inputi,targeti,data_range=torch.max(targeti))
    ssim2 = SSIM(inputi,targeti,data_range=torch.max(targeti))
    print(psnr1,psnr2,ssim1,ssim2)
    if psnr1 > 10 and psnr2 > 10 and psnr1 > psnr2:
        plt.subplot(1,3,1)
        plt.title('Input EM, pnsr=' + str(psnr2))
        plt.axis('off')
        plt.imshow(input[i],cmap="gray",vmin=0,vmax=0.5)
        plt.subplot(1,3,2)
        plt.title('Pred PET, pnsr=' + str(psnr1))
        plt.axis('off')
        plt.imshow(pred[i],cmap="gray",vmin=0,vmax=0.5)
        plt.subplot(1,3,3)
        plt.title('Ground Truth PET')
        plt.axis('off')
        plt.imshow(target[i],cmap="gray",vmin=0,vmax=0.5)
        plt.show()

