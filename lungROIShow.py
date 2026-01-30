import numpy as np
import matplotlib.pyplot as plt

import torch

from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM

from tqdm import *



models = ['EM-U','EM-SAM','EM-SAMF','EM-SAM-A','EM-U-SAM','EM-U-SAM-G','EM-U-P','EM-U-TP','EM-U-SAM-G3CP-GANC']
cmodels = ['CNNBPNet','FBPNet_prior','ADMMNet','IterativeNet']

rkem = np.load('datasetK3-PETCT-RKEM-192T-Ga-Test.npz')['kem'][:,1]
osem = np.load('dataSetK3-CT-OSEM-192-Test.npz')['input'][:,1]

kem = np.load('./dataSetK3-CT-KEM-192-Ga-Test.npz')['input'][:,1]

model_name = models[-1]
data = np.load('result/'+model_name+'.npz')

pred = data['pred'][:,0]
target = data['target'][:,0]
input = data['input'][:,0]
vrange = 1.0

rdata = np.load('./ctpetraw.npz')
lens = rdata['lens']
print(lens,len(pred))

for i in range(0,int(len(lens) * 0.2)):
    total_len = lens[-1-i] - lens[-2-i]
    start = lens[-1] - lens[-2-i]
    end = 1+start-total_len
    print(112-i,total_len,start,end)
    if ( 112 - i == 110): # 110 109
        plt.subplot(1,6,1)
        plt.title('EM')
        plt.axis('off')
        plt.imshow(input[-start:-end,90,:],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(1,6,2)
        plt.title('OSEM')
        plt.axis('off')
        plt.imshow(osem[-start:-end,90,:],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(1,6,3)
        plt.title('KEM-CT-Kernel')
        plt.axis('off')
        plt.imshow(kem[-start:-end,90,:],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(1,6,4)
        plt.title('RKEM')
        plt.axis('off')
        plt.imshow(rkem[-start:-end,90,:],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(1,6,5)
        plt.title('Pred PET')
        plt.axis('off')
        plt.imshow(pred[-start:-end,90,:],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(1,6,6)
        plt.title('Ground Truth PET-'+str(112-i))
        plt.axis('off')
        plt.imshow(target[-start:-end,90,:],cmap="gray",vmin=0,vmax=vrange)
        plt.show()

