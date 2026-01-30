import numpy as np

from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR

import torch


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
        

def metric(pred,target):
    psnr = 0
    ssim = 0
    datLen = len(pred)
    for i in range(datLen):
        # psnr = psnr + PSNR(pred[i],target[i])
        psnr = psnr + PSNR(pred[i],target[i],data_range=torch.max(target[i]))
        # ssim = ssim + PSNR(pred[i],target[i])
        # ssim = ssim + cal_ssim(pred[i],target[i],data_range=1.0)
    psnr = psnr / datLen
    # ssim = ssim / datLen
    return psnr