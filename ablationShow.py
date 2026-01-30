import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM



models = ['EM-U','EM-SAM','EM-SAMF','EM-SAM-A','EM-U-SAM','EM-U-SAM-G']
preds = []
input = []
target = []
for m in models:
    data = np.load('result/'+m+'.npz')
    preds.append(data['pred'])
    if m == models[0]:
        input = data['input'][:,0]
        target = data['target']

total_len = len(input)
# select = [1234,1871,3733,3736,4385,4389,5382,8022,9276,11839]
select = [3677,3785,5375,7212,11744]
# select = [67,562,1114,1131,1253]

vr = 1.0
c = 0
# for i in range(0,total_len):
for i in select:
    print(i)
    inputi = torch.Tensor(input[i]).unsqueeze(0).unsqueeze(0)
    targeti = torch.Tensor(target[i]).unsqueeze(0).unsqueeze(0)
    psnri = PSNR(inputi,targeti,data_range=1.0)

    psnrs = []


    for j in range(len(models)):
        predi = torch.Tensor(preds[j][i]).unsqueeze(0).unsqueeze(0)
        psnrs.append(PSNR(predi,targeti,data_range=1.0))


    if psnrs[-1] - psnrs[0] < 5  and psnrs[-1] - psnrs[0] > 3 and psnrs[-1] >= 10:
        plt.subplot(5,8,c*8+1)
        plt.title('Input EM, pnsr=' + ("%.2f" % psnri.item()))
        plt.axis('off') 
        plt.imshow(input[i],cmap="gray",vmin=0,vmax=vr)

        for k in range(6):
            plt.subplot(5,8,k+2+c*8)
            plt.title(models[k]+',pnsr=' + ("%.2f" % psnrs[k].item()))
            plt.axis('off')
            plt.imshow(preds[k][i][0],cmap="gray",vmin=0,vmax=vr)

        plt.subplot(5,8,c*8+8)
        plt.title('Ground Truth PET')
        plt.axis('off')
        plt.imshow(target[i][0],cmap="gray",vmin=0,vmax=vr)

        c = c + 1

plt.show()

