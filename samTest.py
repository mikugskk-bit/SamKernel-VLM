from argparse import Namespace
import torch
import numpy as np
import cv2
from predictor_sammed import SammedPredictor
from build_sam import sam_model_registry
import matplotlib.pyplot as plt
import scipy.sparse

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


data = np.load('EMData.npy')

image = data[70:73]

plt.imshow(data[10])
plt.show()

# image = pad_and_center_image(image,(256,256))
new_image = np.zeros([256,256,3])
for i in range(len(image)):
    new_image[:,:,i]  = cv2.resize(image[i], (256, 256), interpolation=cv2.INTER_CUBIC)
print(new_image.shape)
image = new_image

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

args = Namespace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "F:\sam-med2d_b.pth"
model = sam_model_registry["vit_b"](args).to(device)
predictor = SammedPredictor(model)
predictor.set_image(image)
ori_h, ori_w, _ = image.shape
input_point = np.array([[162, 127]])
input_label = np.array([1])
plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()