import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM


def normalize_max_min(img,gt):
    Max = np.max(gt)
    Min = np.min(gt)
    return (img - Min) / ((Max - Min) + 1e-6)  #归一化

# data = np.load('./ctpetraw.npz')
data = np.load('./ctpetraw.npz')

kem = np.load('dataSetK3-CT-KEM-192-Ga-Test.npz')['input'][:,1]
rkem = np.load('datasetK3-PETCT-RKEM-192T-Ga-Test.npz')['kem'][:,0]
osem = np.load('dataSetK3-CT-OSEM-192-Test.npz')['input'][:,1]

n_examples = len(data['PET'])
# n_examples = 1000
n_train = int(n_examples * 0.8) 

ct = data['CT'][n_train:]
pet_gt = data['PET'][n_train:]

models = ['CNNBPNet','FBPNet_prior','ADMMNet','EM-U-P','EM-U-SAM-G3CPUF']
preds = []
input = []
target = []
# ct = []
for m in models:
    data = np.load('result/'+m+'.npz')
    preds.append(data['pred'])

    if m == models[-1]:
        input = data['input'][:,0]
        target = data['target']

total_len = len(input)
# select = [1234,1771,3733,3736,4375,4379,5372,7022,9276,11739]
# select = [3677,3775,5375,10212,11744]
# select = [67,562,1114,1131,1253]
# select = [429, 869, 1392, 1394, 1395, 2212, 2668, 2758, 2914, 2929, 2956, 4067, 4091, 4644, 4674, 5083, 5666, 6189]
select = [32, 1395, 2304, 2914, 4067]
# cts = [68, 1395, 2304, 2914, 4067]
vr = 1.0
c = 0

zoom = 12
fsize = 10

zoomX = [88,57,91,66,86]
zoomY = [70,69,68,90,81]

mname = ['CNNBPNet','FBPNet','ADMMNet','EM+Unet','Proposed']

zp = (0.75, 0.0, 0.35, 0.35)
def drawZoom(ax,data,count):
    # zoom in
    axins = ax.inset_axes(zp)
    axins.imshow(data[zoomY[count]-16:zoomY[count]+16,zoomX[count]-16:zoomX[count]+16],cmap='gray',vmin=np.min(data),vmax=np.max(data))
    axins.axis('off')
    # zoom in box
    plt.gca().add_patch(
        plt.Rectangle((zoomX[count]-16, zoomY[count]-16), 32,
                    32, fill=False,
                    edgecolor='r', linewidth=1))  
# for i in range(0,total_len):
for i in select:
    # zoom in count
    # if i == 32 or i == 2304:
    kem[i] = normalize_max_min(kem[i]*30000,pet_gt[i])
    rkem[i] = normalize_max_min(rkem[i]*30000,pet_gt[i])
    osem[i] = normalize_max_min(osem[i]*30000,pet_gt[i])
    if c >= 5:
        c = 0
        plt.show()

    print(i)
    kemi = torch.Tensor(kem[i]).unsqueeze(0).unsqueeze(0)
    rkemi = torch.Tensor(rkem[i]).unsqueeze(0).unsqueeze(0)
    osemi = torch.Tensor(osem[i]).unsqueeze(0).unsqueeze(0)
    targeti = torch.Tensor(target[i]).unsqueeze(0).unsqueeze(0)
    psnri = PSNR(kemi,targeti,data_range=1.0)
    psnri2 = PSNR(rkemi,targeti,data_range=1.0)
    psnri3 = PSNR(osemi,targeti,data_range=1.0)

    psnrs = []


    for j in range(len(models)):
        predi = torch.Tensor(preds[j][i]).unsqueeze(0).unsqueeze(0)
        psnrs.append(PSNR(predi,targeti,data_range=1.0))


    if psnrs[-1] - psnrs[-2] > 2.0 and psnrs[-1] > psnri and psnrs[-1] > psnrs[0] and psnrs[-1] > psnrs[1] and psnrs[-1] > psnrs[2]:
        select.append(i)
        ax = plt.subplot(5,10,c*10+1)
        if c == 0:
            plt.title('CT',family='Times New Roman')
        # plt.text(0,zoom*2 - 1,'PSNR='+str('%.3f' % psnri),fontsize=fsize, color = "r",family='Times New Roman')
        plt.axis('off') 
        plt.imshow(ct[i],cmap="gray")
        drawZoom(ax,ct[i],c)

        ax = plt.subplot(5,10,c*10+2)
        if c == 0:
            plt.title('KEM-CTK',family='Times New Roman')
        plt.text(0,zoom*2 - 1,'PSNR='+str('%.3f' % psnri),fontsize=fsize, color = "r",family='Times New Roman')
        plt.axis('off') 
        plt.imshow(kem[i],cmap="gray",vmin=0,vmax=vr)
        drawZoom(ax,kem[i],c)


        ax = plt.subplot(5,10,c*10+3)
        if c == 0:
            plt.title('RKEM',family='Times New Roman')
        plt.text(0,zoom*2 - 1,'PSNR='+str('%.3f' % psnri2),fontsize=fsize, color = "r",family='Times New Roman')
        plt.axis('off') 
        plt.imshow(rkem[i],cmap="gray",vmin=0,vmax=vr)
        drawZoom(ax,rkem[i],c)

        ax = plt.subplot(5,10,c*10+4)
        if c == 0:
            plt.title('OSEM',family='Times New Roman')
        plt.text(0,zoom*2 - 1,'PSNR='+str('%.3f' % psnri3),fontsize=fsize, color = "r",family='Times New Roman')
        plt.axis('off') 
        plt.imshow(osem[i],cmap="gray",vmin=0,vmax=vr)
        drawZoom(ax,osem[i],c)

        for k in range(5):
            ax = plt.subplot(5,10,1+k+4+c*10)
            if c == 0:
                plt.title(mname[k],family='Times New Roman')
            plt.text(0,zoom*2 - 1,'PSNR='+str('%.3f' % psnrs[k]),fontsize=fsize, color = "r",family='Times New Roman')
            plt.axis('off')
            plt.imshow(preds[k][i][0],cmap="gray",vmin=0,vmax=vr)
            drawZoom(ax,preds[k][i][0],c)

        ax = plt.subplot(5,10,c*10+10)
        if c == 0:
            plt.title('Ground Truth PET',family='Times New Roman')
        plt.axis('off')
        plt.imshow(target[i][0],cmap="gray",vmin=0,vmax=vr)
        drawZoom(ax,target[i][0],c)

        c = c + 1

plt.show()
print(select)

