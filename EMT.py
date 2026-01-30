import numpy as np
import scipy.sparse
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
def normalize(I):
    if isinstance(I,np.ndarray):
        normI = (I-np.amin(I,axis=(-1,-2)))/(np.amax(I,axis=(-1,-2))-np.amin(I,axis=(-1,-2)))
    elif isinstance(I,torch.Tensor):
        normI = (I-torch.amin(I,dim=(-1,-2)))/(torch.amax(I,dim=(-1,-2))-torch.amin(I,dim=(-1,-2)))
    else:
        raise TypeError("I must be cupy.ndarray or numpy.ndarray or torch.Tensor")
    return normI
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
def mynp_divide(a,b,out=None,where=None):
    if where is not None and out is not None:
        return np.where(where,np.divide(a,b),out)
    elif where is None:
        return np.divide(a,b,out)
    else:
        out = np.zeros_like(a)
        return np.where(where,np.divide(a,b),out)
def one_iter(sinogram,mul_factor,pet_noise,G,K,G_row_SumofCol,pet_eval,pet_gt,method_index='KEM',radon=None):
    if K is None:
        K = scipy.sparse.identity(np.prod(pet_gt.shape), format='coo')
    tmp_pet_eval = K.dot(pet_eval)
    if radon is not None:
        l = proj_forward(tmp_pet_eval.reshape(pet_gt.shape),radon)
    else:
        l = G.dot(tmp_pet_eval)
    sinogram_eval = mul_factor*l + pet_noise
    tmp = mynp_divide(sinogram,sinogram_eval+np.mean(sinogram)*1e-9,out=np.ones_like(sinogram_eval))
    tmp[np.logical_and(sinogram==0,sinogram_eval==0)]=1
    if method_index == 'KEM':
        if radon is not None:
            backproj = K.T.dot(proj_backward(mul_factor*tmp,radon))
            pet_eval = pet_eval/(K.T.dot(G_row_SumofCol))*backproj
        else:
            backproj = K.T.dot(G.T.dot(mul_factor*tmp))
            pet_eval = pet_eval/(K.T.dot(G_row_SumofCol))*backproj
    elif method_index == 'MLEM':
        if radon is not None:
            backproj = proj_backward(mul_factor*tmp,radon)
            pet_eval = pet_eval/(G_row_SumofCol)*backproj
        else:
            backproj = G.T.dot(mul_factor*tmp)
            pet_eval = pet_eval/(G_row_SumofCol)*backproj
    else:
        raise ValueError("method_index must be 'KEM' or 'MLEM'")
    pet_eval[G_row_SumofCol==0] = 0
    return pet_eval
def returnResult_MLEMorKEM(sinogram,mul_factor,pet_noise,G,pet_gt,K=None,method_index='KEM',prefix='',radon=None,init=None,total_iter=None):
    if init is None:
        pet_eval = np.ones_like(pet_gt).reshape(-1,1)
    else:
        pet_eval = init.reshape(-1,1)
    if radon is not None:
        G_row_SumofCol = proj_backward(mul_factor,radon)
    else:
        G_row_SumofCol = G.T.dot(mul_factor)
    gt = (normalize(pet_gt)*255).astype(np.uint8)
    if total_iter is None:
        if K is not None:
            total_iter=60
        else:
            total_iter = 25
    if K is None:
        K =scipy.sparse.identity(np.prod(pet_gt.shape), format='coo')
    imgs = [K.dot(pet_eval).reshape(pet_gt.shape)]
    psnrs = [compare_psnr(pet_gt, imgs[0],data_range=np.max(pet_gt))]
    ssims = [compare_ssim(pet_gt, imgs[0],data_range=np.max(pet_gt))]
    mses = [compare_mse(pet_gt, imgs[0])]
    norm_psnrs = [compare_psnr(gt, (normalize(imgs[0])*255).astype(np.uint8))]
    norm_ssims = [compare_ssim(gt, (normalize(imgs[0])*255).astype(np.uint8))]
    norm_mses = [compare_mse(gt, (normalize(imgs[0])*255).astype(np.uint8))]
    for iter in range(total_iter):
        pet_eval = one_iter(sinogram,mul_factor,pet_noise,G,K,G_row_SumofCol,pet_eval,pet_gt,method_index,radon)
        img = K.dot(pet_eval).reshape(pet_gt.shape)
        # img = normalize(img)*np.max(pet_gt)
        # img = img.clip(0,np.max(pet_gt))
        imgs.append(img)
        psnr = compare_psnr(pet_gt, img,data_range=np.max(pet_gt))
        ssim = compare_ssim(pet_gt, img,data_range=np.max(pet_gt))
        mse = compare_mse(pet_gt, img)
        psnrs.append(psnr)
        ssims.append(ssim)
        mses.append(mse)
        img = (normalize(img)*255).astype(np.uint8)
        norm_psnr = compare_psnr(gt, img)
        norm_ssim = compare_ssim(gt, img)
        norm_mse = compare_mse(gt, img)
        norm_psnrs.append(norm_psnr)
        norm_ssims.append(norm_ssim)
        norm_mses.append(norm_mse)
    return np.max(psnrs),np.max(ssims),np.min(mses),imgs[-1]
def read_data(filename):
    import h5py
    data = h5py.File(filename,'r')
    return data
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
import random
from torch_radon import ParallelBeam
import matplotlib.pyplot as plt





if __name__=="__main__":
    # 读取PET数据

    data = np.load('test1.npy')
    EMdata = np.zeros([170,128,128])
    # data1 = read_data("pet_new.mat")
    # pet_gt,mr = np.array(data1["pet_image"]).T,np.array(data1["mri_image"]).T
    # pet_gt,mr = pad_and_center_image(np.stack([pet_gt,mr], axis=0),(128,128))
    for i in range(170):
        pet_gt = pad_and_center_image(data[i],(128,128))

        pet_gt1 = pet_gt
        pet_gt1 = normalize(pet_gt1)*random.uniform(1,8)
        # 投影矩阵
        radon = ParallelBeam(128,np.linspace(0,180,128,False))
        # 从PET图像中仿真正弦图（sinogram）
        sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=20e4)
        # 使用EM生成PET图像
        _,_,_,img = returnResult_MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=20)

        EMdata[i] = img

    np.save('EMData.npy',EMdata)
    
    # 画图
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(pet_gt1,cmap="jet",vmin=0,vmax=np.max(pet_gt1))
    # plt.colorbar()
    # plt.subplot(1,2,2)
    # plt.imshow(img,cmap="jet",vmin=0,vmax=np.max(pet_gt1))
    # plt.show()