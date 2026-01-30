import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM

import torch


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_points(coords, labels, ax, marker_size=150):

    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_mask(mask, ax, random_color=False):
    mask = mask > 0.5
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.3])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


path = './promptResBox'
path2 = './promptResPoint'

files = os.listdir(path)
files2 = os.listdir(path2)
count = 0

ins = [50,55,56,15,124,131,168]
jns = [18,9,13,10,12,6,16]

indexs = []
for i in range(len(ins)):
    indexs.append(ins[i]*24+jns[i])

zoom = 10
fsize = 10
print(indexs)
si = []
sj = []
for i in range(0,len(files),1):
    file = os.path.join(path,files[i])
    file2 = os.path.join(path2,files2[i])
    print(file)

    data = np.load(file)
    data2 = np.load(file2)
    input = data['input']

    sam_mask = data['sam_mask']
    sam_mask2 = data2['sam_mask']
    point= data2['point']
    box = data['box']
    imask = data['imask']
    sam_out = data['sam_out']
    output = data['output']
    output2 = data2['output']
    xo = data['xo']
    gt = data['gt']




    plt.set_cmap('gray')

    for j in range(0,len(gt),1):
        if count == 5:
            print("------------------------------------------------------------------")
            print(si,sj)
            si = []
            sj = []
            plt.show()

            count = 0
        psnr1 = PSNR(torch.Tensor(output[j:j+1]),torch.Tensor(gt[j:j+1]),data_range=1.0)
        psnr2 = PSNR(torch.Tensor(output2[j:j+1]),torch.Tensor(gt[j:j+1]),data_range=1.0)
        ssim1 = SSIM(torch.Tensor(output[j:j+1]),torch.Tensor(gt[j:j+1]),data_range=1.0,kernel_size=11,gaussian_kernel=False)
        print(i,j,psnr1,psnr2)
        index = i * 24 + j

        if index in indexs:
            si.append(i)
            sj.append(j)
            # plt.subplot(5,9,1+count*9)
            # if count == 0:
            #     plt.title('KEM')
            # plt.imshow(xo[j][2])
            # plt.axis('off')

            # ax = plt.subplot(5,9,2+count*9)
            # if count == 0:
            #     plt.title('prompt')
            # plt.imshow(np.swapaxes(np.swapaxes(input[j],0,2),0,1))
            # plt.axis('off')
            # plt.scatter(point[j][:,1],point[j][:,0], s=25, c='r')
            # rect1 = patches.Rectangle((box[j][2], box[j][3]), box[j][0] - box[j][2], box[j][1]-box[j][3], 
            #                         linewidth=1, edgecolor='r',facecolor='none',angle=0)
            # rect2 = patches.Rectangle((box[j][2+4], box[j][3+4]), box[j][0+4] - box[j][2+4], box[j][1+4]-box[j][3+4], 
            #                         linewidth=1, edgecolor='r',facecolor='none',angle=0)
            # rect3 = patches.Rectangle((box[j][2+8], box[j][3+8]), box[j][0+8] - box[j][2+8], box[j][1+8]-box[j][3+8], 
            #                         linewidth=1, edgecolor='r',facecolor='none',angle=0)

            # currentAxis = ax
            # currentAxis.add_patch(rect1)
            # currentAxis.add_patch(rect2)
            # currentAxis.add_patch(rect3)

            for k in range(3):
                ax = plt.subplot(5,9,1+k*1+count*9)
                if count == 0:
                    plt.title('prompt'+str(k),family='Times New Roman')
                plt.imshow(input[j,k])

                # plt.scatter(point[j][:k+1,0],point[j][:k+1,1], s=25, c='r')
                input_label = np.array([1,1,1])

                plt.axis('off')
                # rect1 = patches.Rectangle((box[j][2+4*k], box[j][3+4*k]), box[j][0+4*k] - box[j][2+4*k], box[j][1+4*k]-box[j][3+4*k], 
                #                         linewidth=1, edgecolor='r',facecolor='none',angle=0)
                # ax.add_patch(rect1)
                show_box(box[j][k*4:(k+1)*4],ax)
                show_mask(sam_mask[j][k], plt.gca())

            plt.subplot(5,9,4+count*9)
            if count == 0:
                plt.title('output',family='Times New Roman')
            plt.imshow(output[j][0],vmin=0,vmax=1)
            plt.text(0,zoom*2 - 1,'PSNR='+str('%.3f' % psnr1),fontsize=fsize, color = "r",family='Times New Roman')
            plt.axis('off')

            for k in range(3):
                ax = plt.subplot(5,9,5+k*1+count*9)
                if count == 0:
                    plt.title('prompt'+str(k),family='Times New Roman')
                plt.imshow(input[j,k])

                # plt.scatter(point[j][:k+1,0],point[j][:k+1,1], s=25, c='r')
                input_label = np.array([1,1,1])
                show_points(point[j][:k+1], input_label[:k+1], plt.gca())
                plt.axis('off')
                # rect1 = patches.Rectangle((box[j][2+4*k], box[j][3+4*k]), box[j][0+4*k] - box[j][2+4*k], box[j][1+4*k]-box[j][3+4*k], 
                #                         linewidth=1, edgecolor='r',facecolor='none',angle=0)
                # ax.add_patch(rect1)

                show_mask(sam_mask2[j][k], plt.gca())

            plt.subplot(5,9,8+count*9)
            if count == 0:
                plt.title('output',family='Times New Roman')
            plt.imshow(output2[j][0],vmin=0,vmax=1)
            plt.text(0,zoom*2 - 1,'PSNR='+str('%.3f' % psnr2),fontsize=fsize, color = "r",family='Times New Roman')
            plt.axis('off')


            plt.subplot(5,9,9+count*9)
            if count == 0:
                plt.title('GT',family='Times New Roman')
            plt.imshow(gt[j][0],vmin=0,vmax=1)
            plt.axis('off')

            count = count + 1

            plt.show()


plt.show()
