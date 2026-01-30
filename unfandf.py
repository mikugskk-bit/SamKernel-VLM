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
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


path = './promptRes'
path2 = './promptResUF'

files = os.listdir(path)
files2 = os.listdir(path)
count = 0

ins = [3,4,8,11,14,15,21,28,29,51,52]
jns = [10,20,0,10,20,10,20,20,0,10,10]

indexs = []
for i in range(5,10,1):
    indexs.append(ins[i]*24+jns[i])


print(indexs)

for i in range(0,len(files),1):
    file = os.path.join(path,files[i])
    file2 = os.path.join(path2,files2[i])


    data = np.load(file)
    data2 = np.load(file2)


    sam_mask1 = data['sam_mask']
    sam_mask2 = data2['sam_mask']

 



    plt.set_cmap('gray')

    for j in range(0,len(sam_mask1),1):

        plt.subplot(1,2,1)
        plt.imshow(sam_mask1[j,-1])
        plt.subplot(1,2,2)
        plt.imshow(sam_mask2[j,-1])

        plt.show()


