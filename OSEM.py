import torch
from skimage.data import shepp_logan_phantom
import numpy as np
import scipy

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


def normalize(I):
    if isinstance(I,np.ndarray):
        normI = (I-np.amin(I,axis=(-1,-2)))/(np.amax(I,axis=(-1,-2))-np.amin(I,axis=(-1,-2)))
    elif isinstance(I,torch.Tensor):
        normI = (I-torch.amin(I,dim=(-1,-2)))/(torch.amax(I,dim=(-1,-2))-torch.amin(I,dim=(-1,-2)))
    else:
        raise TypeError("I must be cupy.ndarray or numpy.ndarray or torch.Tensor")
    return normI


def PET2Sinogram(img,radon,count=5e4):
    # count 是计数 用于控制噪声 count高 100e4 就是低噪声 count低20e4 就是低噪声
    if isinstance(img,np.ndarray):
        x = torch.from_numpy(img).float().cuda()
    proj = radon.forward(x)
    mul_factor = torch.ones_like(proj)
    mul_factor = mul_factor+(torch.rand_like(mul_factor)*0.2-0.1)
    noise = torch.ones_like(proj)*torch.mean(mul_factor*proj,dim=(-1,-2),keepdims=True) *0.2
    sinogram = mul_factor*proj + noise
    cs = count/(1e-9+torch.sum(sinogram,dim=(-1,-2),keepdim=True))
    sinogram = sinogram*cs
    mul_factor = mul_factor*cs
    noise = noise*cs
    x = torch.poisson(sinogram)
    return x.detach().cpu().numpy().reshape(-1,1),mul_factor.detach().cpu().numpy().reshape(-1,1),noise.detach().cpu().numpy().reshape(-1,1)


def proj_backward(sinogram,radon=None,G=None):
    assert radon is not None or G is not None,"proj_backward, radon or G must not be None"
    if radon is not None:
        assert np.prod(sinogram.shape)%(radon.volume.height*radon.volume.width)==0,f"sinogram should can reshape to height {radon.volume.height}, width {radon.volume.width},get shape {sinogram.shape}"
        if isinstance(sinogram,np.ndarray):
            out = radon.backward(torch.from_numpy(sinogram).reshape(np.prod(sinogram.shape)//(radon.volume.height*radon.volume.width),radon.volume.height,radon.volume.width).float().cuda()).detach().cpu().numpy()
        elif isinstance(sinogram,torch.Tensor):
            out = radon.backward(sinogram.reshape(np.prod(sinogram.shape)//(radon.volume.height*radon.volume.width),radon.volume.height,radon.volume.width).float().cuda()).detach().cpu().numpy()
        return out.reshape(np.prod(sinogram.shape)//(radon.volume.height*radon.volume.width),-1).T
    elif G is not None:
        assert np.prod(sinogram.shape)%G.shape[0]==0,f"sinogram should can reshape to height {G.shape[0]},get shape {sinogram.shape}"
        if isinstance(sinogram,np.ndarray):
            out = G.T.dot(sinogram.reshape(-1,np.prod(sinogram.shape)//G.shape[0]))
        elif isinstance(sinogram,torch.Tensor):
            out = G.T.dot(sinogram.reshape(-1,np.prod(sinogram.shape)//G.shape[0]).cpu().numpy())
        return out
    else:
        raise ValueError("proj_backward, radon or G must not be None")
    
def proj_forward(img,radon=None,G=None):
    assert radon is not None or G is not None,"proj_forward, radon or G must not be None"
    if radon is not None:
        assert np.prod(img.shape)%(radon.volume.height*radon.volume.width)==0,f"img should can reshape to height {radon.volume.height}, width {radon.volume.width},get shape {img.shape}"
        if isinstance(img,np.ndarray):
            out = radon.forward(torch.from_numpy(img).reshape(np.prod(img.shape)//(radon.volume.height*radon.volume.width),radon.volume.height,radon.volume.width).float().cuda()).detach().cpu().numpy()
        elif isinstance(img,torch.Tensor):
            out = radon.forward(img.reshape(np.prod(img.shape)//(radon.volume.height*radon.volume.width),radon.volume.height,radon.volume.width).float().cuda()).detach().cpu().numpy()
        return out.reshape(np.prod(img.shape)//(radon.volume.height*radon.volume.width),-1).T
    elif G is not None:
        assert np.prod(img.shape)%G.shape[1]==0,f"img should can reshape to height {G.shape[1]},get shape {img.shape}"
        if isinstance(img,np.ndarray):
            out = G.dot(img.reshape(-1,np.prod(img.shape)//G.shape[1]))
        elif isinstance(img,torch.Tensor):
            out = G.dot(img.reshape(-1,np.prod(img.shape)//G.shape[1]).cpu().numpy())
        return out
    else:
        raise ValueError("proj_forward, radon or G must not be None")
def mynp_divide(a,b,out=None,where=None):
    if where is not None and out is not None:
        return np.where(where,np.divide(a,b),out)
    elif where is None:
        return np.divide(a,b,out)
    else:
        out = np.zeros_like(a)
        return np.where(where,np.divide(a,b),out)


# 设定参数
num_angles = 180  # 投影角度
image_size = 400  # 重建图像的大小
num_subsets = 8   # 子集数量
num_iterations = 4 # 迭代次数

det_count = 400

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch_radon import ParallelBeam

# # 创建平行光束对象
# parallel_beam = ParallelBeam(128,np.linspace(0,180,128,False))



# # 生成模拟图像
# true_image = torch.Tensor(shepp_logan_phantom()).reshape(1,1,image_size,image_size).to(device)

# # 计算模拟的投影数据
# projections = parallel_beam.forward(true_image)
# print(projections.shape)

# # 添加噪声
# noise_level = 0.0
# noisy_projections = projections + noise_level * torch.randn_like(projections)

all_angles = np.linspace(0,np.pi,400,False)
subset_radons = []
sub_len = int(len(all_angles) / num_subsets)
for i in range(num_subsets): 
    angles = all_angles[i*sub_len:(i+1)*sub_len]
    sub_radon = ParallelBeam(det_count,angles)
    subset_radons.append(sub_radon)

pet_gt1 = shepp_logan_phantom()
# pet_gt1 = normalize(pet_gt1)
# pet_gt1 = pad_and_center_image(pet_gt1,(image_size,image_size))
# 投影矩阵
radon = ParallelBeam(det_count,all_angles)
# 从PET图像中仿真正弦图（sinogram）
sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=20e4)
# 使用EM生成PET图像
sinogram = torch.tensor(sinogram).float().reshape(pet_gt1.shape)
pet_gt1 = torch.tensor(pet_gt1).float()
mul_factor = torch.tensor(mul_factor).float().reshape(pet_gt1.shape)
pet_noise = torch.tensor(pet_noise).float().reshape(pet_gt1.shape)

import matplotlib.pyplot as plt
# 初始化重建图像
sub_pet_es = []
# # 迭代重建
# for subset in range(num_subsets):
#     sub_sinogram = sinogram[subset*sub_len:(subset+1)*sub_len].reshape(-1,1)
#     sub_pet_gt = pet_gt1[subset*sub_len:(subset+1)*sub_len]
#     sub_mul_factor = mul_factor[subset*sub_len:(subset+1)*sub_len].cuda()
#     sub_pet_noise = pet_noise[subset*sub_len:(subset+1)*sub_len].cuda()
#     print(sinogram.shape,pet_gt1.shape,mul_factor.shape,pet_noise.shape)
#     print(sub_sinogram.shape,sub_pet_gt.shape,sub_mul_factor.shape,sub_pet_noise.shape)

#     sub_pet_eval = torch.ones_like(pet_gt1).reshape(-1,1).cuda()
#     # G_row_SumofCol = proj_backward(sub_mul_factor,subset_radons[subset])
#     G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)
#     for iteration in range(num_iterations):
#         # l = proj_forward(sub_pet_eval.reshape(sub_pet_gt.shape),subset_radons[subset])
#         l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape).cuda())
#         sinogram_eval = (sub_mul_factor.reshape(-1,1) * l.reshape(-1,1) + sub_pet_noise.reshape(-1,1)).detach().cpu()
#         # tmp = mynp_divide(sub_sinogram,sinogram_eval+torch.mean(sub_sinogram)*1e-9,out=np.ones_like(sinogram_eval))
#         tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
#         # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
#         # backproj = proj_backward(sub_mul_factor*tmp,subset_radons[subset])
#         backproj = subset_radons[subset].backward((sub_mul_factor.reshape(-1,1)*tmp.float().cuda()).reshape(sub_mul_factor.shape))
#         sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(-1,1))*backproj.reshape(-1,1)
#         sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
#         print(torch.sum(sub_sinogram - sinogram_eval))

#     sub_pet_es.append(sub_pet_eval.detach().cpu().numpy().reshape(pet_gt1.shape))

# img = np.ones_like(pet_gt1)
# for i in range(num_subsets):
#     img = img * sub_pet_es[i]

# img = img / num_subsets


# 迭代重建
sub_pet_eval = torch.ones_like(pet_gt1).reshape(-1,1).cuda()
for iteration in range(num_iterations):
    for subset in range(num_subsets):
        sub_sinogram = sinogram[subset*sub_len:(subset+1)*sub_len].reshape(-1,1)
        sub_pet_gt = pet_gt1[subset*sub_len:(subset+1)*sub_len]
        sub_mul_factor = mul_factor[subset*sub_len:(subset+1)*sub_len].cuda()
        sub_pet_noise = pet_noise[subset*sub_len:(subset+1)*sub_len].cuda()
        G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

        l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape).cuda())
        sinogram_eval = (sub_mul_factor.reshape(-1,1) * l.reshape(-1,1) + sub_pet_noise.reshape(-1,1)).detach().cpu()
        tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
        tmp[np.logical_and(sub_sinogram.detach().cpu().numpy()==0,sinogram_eval.detach().cpu().numpy()==0)]=1
        backproj = subset_radons[subset].backward((sub_mul_factor.reshape(-1,1)*tmp.float().cuda()).reshape(sub_mul_factor.shape))
        sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(-1,1)
        # sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
        print(torch.sum(sub_sinogram - sinogram_eval))

    sub_pet_es.append(sub_pet_eval.detach().cpu().numpy().reshape(pet_gt1.shape))

img = sub_pet_eval.detach().cpu().numpy().reshape(pet_gt1.shape)


import matplotlib.pyplot as plt
plt.subplot(1,4,1)
plt.imshow(pet_gt1.cpu().numpy(), cmap='gray')
plt.title('Input Image')
plt.subplot(1,4,2)
plt.imshow(sinogram.cpu().numpy().reshape(pet_gt1.shape), cmap='gray')
plt.title('Projection')
plt.subplot(1,4,3)
plt.imshow(proj_backward(sinogram,radon).reshape(pet_gt1.shape), cmap='gray')
plt.title('Noisy Projection')
plt.subplot(1,4,4)
plt.imshow(img, cmap='gray')
plt.title('Reconstructed Image')
plt.show()

