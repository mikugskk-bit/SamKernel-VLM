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
        plt.subplot(3,6,1)
        plt.title('EM')
        plt.axis('off')
        plt.imshow(input[-start+45:-start+45+50,90,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,2)
        plt.title('OSEM')
        plt.axis('off')
        plt.imshow(osem[-start+45:-start+45+50,90,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,3)
        plt.title('KEM-CT-Kernel')
        plt.axis('off')
        plt.imshow(kem[-start+45:-start+45+50,90,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,4)
        plt.title('RKEM')
        plt.axis('off')
        plt.imshow(rkem[-start+45:-start+45+50,90,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,5)
        plt.title('Pred PET')
        plt.axis('off')
        plt.imshow(pred[-start+45:-start+45+50,90,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,6)
        plt.title('Ground Truth PET-'+str(112-i))
        plt.axis('off')
        plt.imshow(target[-start+45:-start+45+50,90,50:100],cmap="gray",vmin=0,vmax=vrange)

        plt.subplot(3,6,1+6)
        plt.axis('off')
        plt.imshow(input[-start+45:-start+45+50,65:115,60],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,2+6)
        plt.axis('off')
        plt.imshow(osem[-start+45:-start+45+50,65:115,60],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,3+6)
        plt.axis('off')
        plt.imshow(kem[-start+45:-start+45+50,65:115,60],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,4+6)
        plt.axis('off')
        plt.imshow(rkem[-start+45:-start+45+50,65:115,60],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,5+6)
        plt.axis('off')
        plt.imshow(pred[-start+45:-start+45+50,65:115,60],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,6+6)
        plt.axis('off')
        plt.imshow(target[-start+45:-start+45+50,65:115,60],cmap="gray",vmin=0,vmax=vrange)

        plt.subplot(3,6,1+12)
        plt.axis('off')
        plt.imshow(input[-start+70,65:115,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,2+12)
        plt.axis('off')
        plt.imshow(osem[-start+70,65:115,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,3+12)
        plt.axis('off')
        plt.imshow(kem[-start+70,65:115,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,4+12)
        plt.axis('off')
        plt.imshow(rkem[-start+70,65:115,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,5+12)
        plt.axis('off')
        plt.imshow(pred[-start+70,65:115,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.subplot(3,6,6+12)
        plt.axis('off')
        plt.imshow(target[-start+70,65:115,50:100],cmap="gray",vmin=0,vmax=vrange)
        plt.show()



        inputv = input[-start+45:-start+45+50,65:115,50:100].reshape(-1) * 30000
        osemv = osem[-start+45:-start+45+50,65:115,50:100].reshape(-1) * 30000
        kemv = kem[-start+45:-start+45+50,65:115,50:100].reshape(-1) * 30000
        rkemv = rkem[-start+45:-start+45+50,65:115,50:100].reshape(-1) * 30000
        predv = pred[-start+45:-start+45+50,65:115,50:100].reshape(-1) * 30000
        targetv = target[-start+45:-start+45+50,65:115,50:100].reshape(-1) * 30000

        fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(16, 3))
        parts = ax2.violinplot([inputv,osemv,kemv,rkemv,predv,targetv], showmeans=True, showmedians=False,
                showextrema=False)
        plt.ylim([0.0,29000])
        plt.show()