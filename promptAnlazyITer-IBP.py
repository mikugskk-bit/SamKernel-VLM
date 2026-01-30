import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as SSIM

import torch


colors = ['red','orange','yellow','green','blue','purple','pink','brown']


def normalize_max_min(img):
    Max = np.max(img)
    Min = np.min(img)
    return (img - Min) / ((Max - Min) + 1e-6)  #归一化

def show_box(box, ax,c):
    print(box)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_points(coords, labels, ax, marker_size=150,index=0):

    print(coords)
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    for i in range(pos_points.shape[0]):
        plt.text(pos_points[i, 0], pos_points[i, 1], str(index), color='red')
    for i in range(pos_points.shape[0]-1):
        plt.plot([pos_points[i, 0],pos_points[i+1, 0]], [pos_points[i, 1],pos_points[i+1, 1]], 'bo', linestyle="--")

def show_mask(mask, ax, random_color=False):
    mask = mask > 0.5
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.3])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

import cv2


path = './promptResPoint-8E-1'

files = os.listdir(path)
count = 0

select = [46,50,75,77,82,30]

select = [376,40,66,80,85,120,172]
selectj = [7,6,1,3,7,5,4]

select = [17,52,75,95,95,146,157,227,229,6,429,471,492,510]
selectj = [4,7, 6, 6, 7,  2,  5 ,4  ,5  ,7,6  ,5  ,4  ,7]
c = 0
# for i in range(0,800,1):
for i in select:
    data = np.load(os.path.join(path,files[i]))

    input = data['input']

    sam_mask = data['sam_mask']
    # sam_mask2 = data['sam_mask2']
    point= data['point']

    box = data['box']
    imask = data['imask']
    sam_out = data['sam_out']
    output = data['output']
    xo = data['xo']
    gt = data['gt']

    print(xo.shape)

    input_label = np.array([1,1,1,1,1,1,1,1])
    # input_label = np.array([1,1,1,1,1])
    # input_label = np.array([1,1,1])

    print(box.shape)

    j = selectj[count]
    # j = 4
    count = count + 1

    for p in range(8):
        if p == 0 :
            xo[j][p] = cv2.GaussianBlur(xo[j][p],(3,3),0.1,1)
        if p == 1 :
            xo[j][p] = cv2.GaussianBlur(xo[j][p],(3,3),0.2,1)
        if p == 2 :
            xo[j][p] = cv2.GaussianBlur(xo[j][p],(3,3),0.5,1)
        if p == 3 :
            xo[j][p] = cv2.GaussianBlur(xo[j][p],(5,5),0.5,1)
        if p == 4 :
            xo[j][p] = cv2.GaussianBlur(xo[j][p],(5,5),1,1)
        if p == 5 :
            xo[j][p] = cv2.GaussianBlur(xo[j][p],(5,5),1.5,1)
        if p == 6 :
            xo[j][p] = cv2.GaussianBlur(xo[j][p],(11,11),1.5,1)
        if p == 7 :
            xo[j][p] = cv2.GaussianBlur(xo[j][p],(11,11),2,1)
        plt.subplot(5,10,1+c*10+p)
        plt.title(str(i)+'-'+str(p))
        plt.imshow(cv2.resize(xo[j][p],(256,256),interpolation=cv2.INTER_NEAREST),cmap='gray',vmin=0,vmax=1)
        # plt.imshow(sam_mask[j][p],cmap='gray',vmin=0,vmax=1)
        plt.axis('off')



        # print(box[j],len(box[j])/4)
        # for k in range(len(box[j])//4):
        show_box(np.clip(box[j][p*4:p*4+4],0,256),plt.gca(),p)
        show_points(point[j][p:p+1], input_label[p:p+1], plt.gca(),index=p)
        print(point[j],i)

    plt.subplot(5,10,1+c*10+8)
    plt.title('Output')
    plt.imshow(cv2.resize(output[j,0],(256,256),interpolation=cv2.INTER_NEAREST),cmap='gray',vmin=0,vmax=1)
    plt.axis('off')

    plt.subplot(5,10,1+c*10+9)
    plt.title('GT')
    plt.imshow(cv2.resize(gt[j,0],(256,256),interpolation=cv2.INTER_NEAREST),cmap='gray',vmin=0,vmax=1)
    plt.axis('off')
    c = c+1

    if c == 5:
        c = 0
        plt.show()

plt.show()