import matplotlib.pyplot as plt
import numpy as np
from matplotlib. patches import Rectangle

data2 = np.load('./datasetB5N-KSAM.npz')['kem']


data  = np.load('promptResPoint-VL/batch-3.npz')


input = data['input']
box = data['box']
output = data['output']
gt = data['gt']
mask = data['imask']
print(box[0])
data3 = data2[len(input)*3:len(input)*4]

for i in range(len(box)):
    print(i)
    fig = plt.figure()
    ax = fig.add_subplot(161)
    plt.imshow(input[i,0],cmap='gray')
    ax = fig.add_subplot(162)
    plt.imshow(input[i,0],cmap='gray')
    ax. add_patch (Rectangle((box[i][0], box[i][1]), box[i][2]- box[i][0], box[i][3]- box[0][1],
                edgecolor='red',
                fill= False, 
                lw= 5 ))
    ax = fig.add_subplot(163)
    plt.imshow(output[i,0],cmap='gray')
    ax = fig.add_subplot(164)
    plt.imshow(gt[i,0],cmap='gray')
    ax = fig.add_subplot(165)
    plt.imshow(input[i,0],cmap='jet')
    ax = fig.add_subplot(166)
    plt.imshow(data3[i,2],cmap='gray')
    plt.show()