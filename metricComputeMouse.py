import numpy as np
import matplotlib.pyplot as plt

import torch

from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as LPIPS
from torchmetrics.functional.image.rmse_sw import root_mean_squared_error_using_sliding_window as RMSE
from torchmetrics.functional.image.uqi import universal_image_quality_index as UIQI

from tqdm import *


# models = ['EM-U','EM-SAM','EM-SAMF','EM-SAM-A','EM-U-SAM','EM-U-SAM-G','EM-U-P','EM-U-TP','EM-U-SAM-G3CP','EM-U-SAM-G3CPUF']
models = ['CNNBPNet','FBPNet_prior','ADMMNet','EM-U-SAM-G3CP']
model_name = models[2]
data = np.load('result/'+model_name+'-mouse.npz')

data = np.load('./mouseRes.npz')


pred = data['pred'][:,0]
target = data['target']
# input = data['input'][:,0]


print(pred.shape)
print(target.shape)

total_psnr = 0
total_ssim = 0
total_lpip = 0
total_rmse = 0
total_uiqi = 0

qbar = trange(len(pred))
for i in range(len(pred)):
    predi = torch.Tensor(pred[i]).unsqueeze(0).unsqueeze(0)
    targeti = torch.Tensor(target[i]).unsqueeze(0).unsqueeze(0)


    psnr1 = PSNR(predi,targeti,data_range=1.0)
    ssim1 = SSIM(predi,targeti,data_range=1.0,kernel_size=11,gaussian_kernel=False)
    rmse1 = RMSE(predi,targeti)
    # lpip1 = LPIPS(predi.repeat(1,3,1,1),targeti.repeat(1,3,1,1),normalize=True)
    uiqi1 = UIQI(predi,targeti)
    total_psnr += psnr1
    total_ssim += ssim1
    # total_lpip += lpip1
    total_rmse += rmse1
    total_uiqi += uiqi1
    qbar.update(1)



qbar.close()
total_psnr = total_psnr / len(pred)
total_ssim = total_ssim / len(pred)
total_lpip = total_lpip / len(pred)
total_rmse = total_rmse / len(pred)
total_uiqi = total_uiqi / len(pred)

print(total_psnr.item(),total_ssim.item(),total_lpip,total_rmse.item(),total_uiqi.item())
