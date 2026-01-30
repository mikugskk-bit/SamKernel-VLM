import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM

import torch


def normalize_max_min(img):
    Max = np.max(img)
    Min = np.min(img)
    return (img - Min) / ((Max - Min) + 1e-6)  #归一化

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


path = './promptRes'

files = os.listdir(path)
count = 0

# ins = [3,4,8,11,14,15,21,28,29,51,52]
# jns = [10,20,0,10,20,10,20,20,0,10,10]

# ins = [16,22,65,85,124,135,203,230,257]
# jns = [6,18,1,17,2,7,1,21,16]

#Freeze And UnFreeze
ins = [16,22,65,85,135,203,230,257]
jns = [6,18,1,17,7,1,21,16]

indexs = []
for i in range(len(ins)):
    indexs.append(ins[i]*24+jns[i])


print(indexs)
i_s = []
j_s = []


ars = [[0.478,0.473,0.465],
       [0.478,0.47,0.465],
       [0.478,0.473,0.465],
       [0.478,0.47,0.465],
       [0.476,0.471,0.465]]


ads = [[0.48,0.47,0.465],
       [0.48,0.47,0.465],
       [0.48,0.47,0.465],
       [0.48,0.47,0.465],
       [0.48,0.47,0.465]]
for i in range(0,len(files),1):
# for i in ins:
    file = os.path.join(path,files[i])
    print(file)

    data = np.load(file)
    input = data['input']

    sam_mask = data['sam_mask']
    sam_mask2 = data['sam_mask2']
    point= data['point']
    box = data['box']
    imask = data['imask']
    sam_out = data['sam_out']
    output = data['output']
    xo = data['xo']
    gt = data['gt']




    plt.set_cmap('gray')

    for j in range(0,len(gt),1):
        if count == 5:
            print("------------------------------------------------------------------")
            print(i_s,j_s)
            i_s = []
            j_s = []
            plt.show()
            count = 0
        psnr1 = PSNR(torch.Tensor(output[j:j+1]),torch.Tensor(gt[j:j+1]),data_range=1.0)
        ssim1 = SSIM(torch.Tensor(output[j:j+1]),torch.Tensor(gt[j:j+1]),data_range=1.0,kernel_size=11,gaussian_kernel=False)
        print(i,j,psnr1,ssim1)
        index = i * 24 + j

        diff1 = np.abs(np.sum(sam_mask2[j][1] - sam_mask2[j][0]))
        diff2 = np.abs(np.sum(sam_mask2[j][2] - sam_mask2[j][1]))

        diffp1 = np.abs(np.sum(point[j][0] - point[j][1]))
        diffp2 = np.abs(np.sum(point[j][1] - point[j][2]))

        print(np.sum(sam_mask2[j][1] - sam_mask2[j][0]),np.sum(sam_mask2[j][2] - sam_mask2[j][1]))
                       

        # if psnr1 > 30 and index in indexs and diff1 >5 and diff2 >5 and diffp1 > 0 and diffp2 > 0:
        if True:
            i_s.append(i)
            j_s.append(j)
            plt.subplot(5,10,1+count*10)
            if count == 0:
                plt.title('EM Input')
            plt.imshow(xo[j][0])
            plt.axis('off')

            # ax = plt.subplot(5,10,2+count*10)
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
            print(box[j],point[j])
            # currentAxis = ax
            # currentAxis.add_patch(rect1)
            # currentAxis.add_patch(rect2)
            # currentAxis.add_patch(rect3)
            # print(np.sum(sam_mask2[j][1] - sam_mask2[j][0]),np.sum(sam_mask2[j][2] - sam_mask2[j][1]))
            for k in range(3):
                ax = plt.subplot(5,10,2+k*2+count*10)
                if count == 0:
                    plt.title('prompt'+str(k))
                plt.imshow(input[j,k])

                # plt.scatter(point[j][:k+1,0],point[j][:k+1,1], s=25, c='r')
                input_label = np.array([1,1,1])
                # show_points(point[j][:k+1], input_label[:k+1], plt.gca())
                plt.axis('off')
                # rect1 = patches.Rectangle((box[j][2+4*k], box[j][3+4*k]), box[j][0+4*k] - box[j][2+4*k], box[j][1+4*k]-box[j][3+4*k], 
                #                         linewidth=1, edgecolor='r',facecolor='none',angle=0)
                # ax.add_patch(rect1)
                # show_box(box[j][k*4:(k+1)*4],ax)
                # show_mask(sam_mask[j][k], plt.gca())
                plt.imshow(sam_mask[j][k],vmin=0.0)


                plt.subplot(5,10,2+k*2+1+count*10)
                if count == 0:
                    plt.title('unfreeze mask'+str(k))
                # if k > 0:
                #     plt.imshow(sam_mask2[j][k] - sam_mask2[j][k-1])
                # else:
                # ma = np.float16(sam_mask2[j][k] > ars[count][k])*0.025+1

                plt.imshow(sam_mask2[j][k],vmin = 0.0)
                plt.axis('off')
            plt.subplot(5,10,8+count*10)
            if count == 0:
                plt.title('mask-adapt')
            plt.imshow(np.swapaxes(np.swapaxes(sam_out[j],0,2),0,1))
            plt.axis('off')
            plt.subplot(5,10,9+count*10)
            if count == 0:
                plt.title('output')
            plt.imshow(output[j][0],vmin=0,vmax=1)
            plt.axis('off')


            plt.subplot(5,10,10+count*10)
            if count == 0:
                plt.title('GT')
            plt.imshow(gt[j][0],vmin=0,vmax=1)
            plt.axis('off')

            count = count + 1

            # plt.show()


plt.show()
