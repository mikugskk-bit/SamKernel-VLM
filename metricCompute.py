import numpy as np
import matplotlib.pyplot as plt

import torch

from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity as LPIPS
from torchmetrics.functional.image.rmse_sw import root_mean_squared_error_using_sliding_window as RMSE
from torchmetrics.functional.image.uqi import universal_image_quality_index as UIQI

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from tqdm import *

import scipy


def pad_and_center_image(images, size):
    h, w = images.shape[-2:]
    modified_size = max(*size, h, w)
    dh = modified_size - h
    dw = modified_size - w

    # 计算补零的位置
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    # 补零
    assert top >= 0 and bottom >= 0 and left >= 0 and right >= 0, 'dh={}, dw={}, h={}, w={}'.format(dh, dw, h, w)
    if len(images.shape)==3:
        padded_image = np.zeros((images.shape[0],modified_size,modified_size))
        padded_image = padded_image+images[:,0:1,0:1]
        if top>=0 and bottom>0 and left>=0 and right>0:
            padded_image[:,top:-bottom,left:-right] = images
        elif top==0 and bottom==0 and left>=0 and right>0:
            padded_image[:,:,left:-right] = images
        elif left==0 and right==0 and top>=0 and bottom>0:
            padded_image[:,top:-bottom,:] = images
        elif top==0 and bottom==0 and left==0 and right==0:
            padded_image = images
        total_imgs = padded_image
        total_imgs = scipy.ndimage.zoom(total_imgs, (1,(size[0])/modified_size,(size[1])/modified_size),order=1)
    elif len(images.shape)==2:
        padded_image = np.zeros((modified_size,modified_size))
        padded_image = padded_image+images[0:1,0:1]
        if top>=0 and bottom>0 and left>=0 and right>0:
            padded_image[top:-bottom,left:-right] = images
        elif top==0 and bottom==0 and left>=0 and right>0:
            padded_image[:,left:-right] = images
        elif left==0 and right==0 and top>=0 and bottom>0:
            padded_image[top:-bottom,:] = images
        elif top==0 and bottom==0 and left==0 and right==0:
            padded_image = images
        total_imgs = padded_image
        total_imgs = scipy.ndimage.zoom(total_imgs, ((size[0])/modified_size,(size[1])/modified_size),order=1)

    assert total_imgs.shape[-2:] == size, 'total_imgs.shape={}, size={}'.format(total_imgs.shape, size)
    return total_imgs

models = ['EM-U','EM-SAM','EM-SAMF','EM-SAM-A','EM-U-SAM','EM-U-SAM-G','EM-U-P','EM-U-TP','EM-U-SAM-G3CP','EM-U-SAM-G3CPD-BNT2','EM-U-SAM-G3CPUF','EM-U-SAM-G3CP-Points','EM-U-SAM-G3CP-Boxs','EM-U-SAM-G3CP-Ga']
model_name = models[1]
cmodels = ['CNNBPNet-B','FBPNet_prior-B','ADMMNet-B','IterativeNet']
imodels = ['OSEM-SAM-3','OSEM-SAM-5','OSEM-SAM-8','OSEM-SAM-single-point','OSEM-SAM-single-box','OSEM-SAM-single-boxpoint','EM-SAM-New','EM-SAM-New-Br','EM-SAM-New-i1']
model_name = cmodels[2]
# model_name = 'EM-SAM-New-Br'
# model_name = 'SAM-decoder-Unf'
model_name = 'SAM-decoder-Unf50HE'
# model_name = 'OSEM'
# model_name = 'ADMMNet-B200000HE'
# model_name = 'EM-SAM-New-Br-NoUnet-unf-s5'
# model_name = 'OSEM'
# model_name = 'RKEM'
# model_name = 'EM-SAM-New-Br-Old100'
# model_name = 'KSAM'
# model_name = 'SAM-decoder-Unf100'

# model_name = 'CNNBPNet-B200000HE'
def normalize(I):
    if isinstance(I,np.ndarray):
        normI = (I-np.amin(I,axis=(-1,-2)))/(np.amax(I,axis=(-1,-2))-np.amin(I,axis=(-1,-2)))
    elif isinstance(I,torch.Tensor):
        normI = (I-torch.amin(I,dim=(-1,-2)))/(torch.amax(I,dim=(-1,-2))-torch.amin(I,dim=(-1,-2)))
    else:
        raise TypeError("I must be cupy.ndarray or numpy.ndarray or torch.Tensor")
    return normI

def normalize_max_min(img,gt):
    Max = np.max(gt)
    Min = np.min(gt)
    return (img - Min) / ((Max - Min) + 1e-6)  #归一化

c=0
def compare_mae(img1,img2,data_range=None):
    return np.mean(np.abs(img1-img2))

if model_name == 'KSAM':
    # data = np.load('result/SAM-decoder-Unf.npz')
    # target = data['target'][:100,0]
    # input = data['input'][:,0]
    # kdata = np.load('datasetB3N-KSAM.npz')['kem'][:,1]
    # pred = kdata
    pred = np.load('datasetB5N-KSAM.npz')['kem'][:,1]
    test_id = np.load('./test_list1.npy')
    target = np.load('/mnt/c/brainweb/pet_vol_slice_144by144_nt.npy')[test_id]
elif model_name == 'OSEM':
    # data = np.load('result/SAM-decoder-Unf.npz')

    # target = data['target'][:,0]
    # input = data['input'][:,0]
    # kdata = np.load('datasetB3N-KSAM.npz')['kem'][:,1]

    target = np.load('/mnt/d/HECKTOR-test.npy')
    pred = np.load('newDataset/osem-h-20e4-30-1-test.npz')['osem']
elif model_name == 'MLEM':
    # kdata = np.load('datasetB3N-KSAM.npz')['kem'][:,1]
    pred = np.load('datasetB5N-EM-test.npz')['kem'][:,0]
    test_id = np.load('./test_list1.npy')
    target = np.load('/mnt/c/brainweb/pet_vol_slice_144by144_nt.npy')[test_id]
    # pred = np.load('datasetH3N-EM-test.npz')['kem'][:,1]
    # target = np.load('/mnt/d/HECKTOR-test.npy')
elif model_name == 'KEM':
    # kdata = np.load('datasetB3N-KSAM.npz')['kem'][:,1]
    pred = np.load('datasetB5N-KEM-test.npz')['kem'][:,4]
    test_id = np.load('./test_list1.npy')
    target = np.load('/mnt/c/brainweb/pet_vol_slice_144by144_nt.npy')[test_id]
    # pred = np.load('datasetH3N-KEM-test.npz')['kem'][:1]
    # target = np.load('/mnt/d/HECKTOR-test.npy')
elif model_name == 'RKEM':
    # kdata = np.load('datasetB3N-KSAM.npz')['kem'][:,1]
    pred = np.load('datasetB5N-RKEM-test.npz')['kem'][:,4]
    test_id = np.load('./test_list1.npy')
    target = np.load('/mnt/c/brainweb/pet_vol_slice_144by144_nt.npy')[test_id]
    # pred = np.load('datasetH3N-RKEM-test.npz')['kem'][:,1]
    # target = np.load('/mnt/d/HECKTOR-test.npy')
else:
    data = np.load('result/'+model_name+'.npz')
    pred = data['pred'][:,0]
    target = data['target'][:,0]
    input = data['input'][:,0]
# pred = data['pred'][:,0,16:176,16:176]
# target = data['target'][:,0,16:176,16:176]
# input = data['input'][:,0,16:176,16:176]

# print(input.shape)
print(pred.shape)
print(target.shape)

print(np.max(pred),np.min(pred))
print(np.max(target),np.min(target))

total_psnr = 0
total_ssim = 0
total_mae = 0
total_rmse = 0
total_uiqi = 0

#
# data = np.load('./ctpetraw.npz')
# n_examples = len(data['PET'])
# n_train = int(n_examples * 0.8) 

# pet_gt = data['PET'][n_train:]
# target = pet_gt

qbar = trange(len(pred))
for i in range(len(pred)):

    #
    # pred[i] = normalize_max_min(pred[i]*30000,pet_gt[i])
    # target[i] = normalize(target[i])

    # if np.sum(target[i]>1)>20:
    #     continue

    # if np.sum(target[i])<100:
    #     continue
    
    predi = torch.Tensor(pred[i]*np.max(target[i])).unsqueeze(0).unsqueeze(0)
    targeti = torch.Tensor(pad_and_center_image(target[i],(128,128))).unsqueeze(0).unsqueeze(0)
    # targeti = torch.Tensor(target[i]).unsqueeze(0).unsqueeze(0)
    # inputi = torch.Tensor(input[i]).unsqueeze(0).unsqueeze(0)

    psnr1 = PSNR(predi,targeti,data_range=np.max(target[i]))
    # psnr1 = compare_psnr(target[i],pred[i].clip(0,np.max(target[i])),data_range=np.max(target[i]))
    mae = compare_mae(pad_and_center_image(target[i],(128,128)),pred[i]*np.max(target[i]),data_range=np.max(target[i]))
    # mae = compare_mae(target[i],pred[i],data_range=np.max(target[i]))
    ssim1 = SSIM(predi,targeti,data_range=np.max(target[i]),kernel_size=11,gaussian_kernel=False)
    rmse1 = RMSE(predi,targeti)
    # lpip1 = LPIPS(predi.repeat(1,3,1,1),targeti.repeat(1,3,1,1),normalize=True)
    uiqi1 = UIQI(predi,targeti)
    if torch.isnan(uiqi1):
        uiqi1 = 0
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
