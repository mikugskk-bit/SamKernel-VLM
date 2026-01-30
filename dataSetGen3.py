import numpy as np
import scipy.sparse
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
from EM import MLEMorKEM, normalize
from tqdm import *
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
from multiprocessing import Pool
import multiprocessing

# if __name__=="__main__":


#     # path = 'F:\Codes\pet_vol_slice_144by144_bwpm.npy'
#     path = './totalbodyDataset/rawdata.npy'

#     # path = 'test1.npy'

#     data = np.load(path)

#     pet_gts = np.zeros((len(data),1,192,192))
#     kemresults = np.zeros((len(data),3,192,192))

#     qbar = trange(len(data))
#     for i in range(len(data)):

#         pet_gt1= pad_and_center_image(data[i],(192,192))
#         if np.sum(pet_gt1) == 0:
#             print('pass')
#             continue
#         pet_gt1 = normalize(pet_gt1)*random.uniform(1,8)
#         # 投影矩阵
#         radon = ParallelBeam(192,np.linspace(0,180,192,False))
#         # 从PET图像中仿真正弦图（sinogram）
#         sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=20e4)

#         sinogram = torch.Tensor(sinogram)
#         mul_factor = torch.Tensor(mul_factor)
#         pet_noise = torch.Tensor(pet_noise)


#         # kemresult1 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=10)
#         # kemresult2 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=20)
#         kemresult1,kemresult2,kemresult3 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=30)

#         pet_gts[i][0] = pet_gt1
#         kemresults[i][0] = kemresult1
#         kemresults[i][1] = kemresult2
#         kemresults[i][2] = kemresult3
#         # pets.append(pet_gt)
#         qbar.update(1)
from scipy.sparse import coo_matrix
def gaussian_kernel(X,kernel_size, sigma, J):

    x_hat = X.reshape(-1, 1)
    rows = []
    cols = []
    data = []

    for i in range(kernel_size):
        # 计算当前位置的邻域
        for j in range(-J, J+1):
            for k in range(-J, J+1):
                row = i // 128 + j
                col = i % 128 + k
                # 确保索引在矩阵范围内
                if 0 <= row < 128 and 0 <= col < 128:
                    dist = np.abs(j) + np.abs(k)
                    if dist <= J:
                        index = row * 128 + col
                        rows.append(i)
                        cols.append(index)
                        data.append(np.exp(-(x_hat[i] - x_hat[index])**2 / (2 * sigma**2)))
    # 确保数组是一维的
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data).reshape(-1)
    # print(rows.shape,cols.shape,data.shape)
    # 创建稀疏矩阵
    kernel = coo_matrix((data, (rows, cols)), shape=(kernel_size, kernel_size))
    return kernel

def gaussian_kernel2(X,Y,kernel_size, sigma, J):

    x_hat = X.reshape(-1, 1)
    y_hat = Y.reshape(-1, 1)
    rows = []
    cols = []
    data = []

    for i in range(kernel_size):
        # 计算当前位置的邻域
        for j in range(-J, J+1):
            for k in range(-J, J+1):
                row = i // 128 + j
                col = i % 128 + k
                # 确保索引在矩阵范围内
                if 0 <= row < 128 and 0 <= col < 128:
                    dist = np.abs(j) + np.abs(k)
                    if dist <= J:
                        index = row * 128 + col
                        rows.append(i)
                        cols.append(index)
                        data.append(np.exp(-(x_hat[i] - y_hat[index])**2 / (2 * sigma**2)))
    # 确保数组是一维的
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data).reshape(-1)
    # print(rows.shape,cols.shape,data.shape)
    # 创建稀疏矩阵
    kernel = coo_matrix((data, (rows, cols)), shape=(kernel_size, kernel_size))
    return kernel


def OSEM(sinogram,mul_factor,pet_noise,pet_gt1,num_subsets,total_iter,det_count):
    sinogram = torch.Tensor(sinogram)
    mul_factor = torch.Tensor(mul_factor)
    pet_noise = torch.Tensor(pet_noise)
    pet_gt1 = torch.Tensor(pet_gt1)
    
    sinogram = sinogram.reshape(pet_gt1.shape)
    mul_factor = mul_factor.reshape(pet_gt1.shape)
    pet_noise = pet_noise.reshape(pet_gt1.shape)

    all_angles = np.linspace(0,np.pi,128,False)
    subset_radons = []
    sub_len = int(len(all_angles) / num_subsets)
    for i in range(num_subsets): 
        angles = all_angles[i*sub_len:(i+1)*sub_len]
        sub_radon = ParallelBeam(det_count,angles)
        subset_radons.append(sub_radon)

    # # 初始化重建图像
    # sub_pet_es = []
    # # 迭代重建
    # for subset in range(num_subsets):
    #     sub_sinogram = sinogram[subset*sub_len:(subset+1)*sub_len].reshape(-1,1)
    #     sub_pet_gt = pet_gt1[subset*sub_len:(subset+1)*sub_len]
    #     sub_mul_factor = mul_factor[subset*sub_len:(subset+1)*sub_len].cuda()
    #     sub_pet_noise = pet_noise[subset*sub_len:(subset+1)*sub_len].cuda()
    #     # print(sinogram.shape,pet_gt1.shape,mul_factor.shape,pet_noise.shape)
    #     # print(sub_sinogram.shape,sub_pet_gt.shape,sub_mul_factor.shape,sub_pet_noise.shape)

    #     sub_pet_eval = torch.ones_like(pet_gt1).reshape(-1,1).cuda()
    #     # G_row_SumofCol = proj_backward(sub_mul_factor,subset_radons[subset])
    #     G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)
    #     for iteration in range(total_iter):
    #         # l = proj_forward(sub_pet_eval.reshape(sub_pet_gt.shape),subset_radons[subset])
    #         l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape).cuda())
    #         sinogram_eval = (sub_mul_factor.reshape(-1,1) * l.reshape(-1,1) + sub_pet_noise.reshape(-1,1)).detach().cpu()
    #         # tmp = mynp_divide(sub_sinogram,sinogram_eval+torch.mean(sub_sinogram)*1e-9,out=np.ones_like(sinogram_eval))
    #         tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9+1e-10)
    #         # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
    #         # backproj = proj_backward(sub_mul_factor*tmp,subset_radons[subset])
    #         backproj = subset_radons[subset].backward((sub_mul_factor.reshape(-1,1)*tmp.float().cuda()).reshape(sub_mul_factor.shape))
    #         sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(-1,1))*backproj.reshape(-1,1)
    #         sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
    #     sub_pet_es.append(sub_pet_eval.detach().cpu().numpy().reshape(pet_gt1.shape))

    # img = np.zeros_like(pet_gt1)
    # for i in range(num_subsets):
    #     img = np.add(img,sub_pet_es[i])

    # img = img / num_subsets

    # 迭代重建
    sub_pet_es= []
    sub_pet_eval = torch.ones_like(pet_gt1).reshape(-1,1).cuda()
    for iteration in range(total_iter):
        for subset in range(num_subsets):
            sub_sinogram = sinogram[subset*sub_len:(subset+1)*sub_len].reshape(-1,1)
            sub_pet_gt = pet_gt1[subset*sub_len:(subset+1)*sub_len]
            sub_mul_factor = mul_factor[subset*sub_len:(subset+1)*sub_len].cuda()
            sub_pet_noise = pet_noise[subset*sub_len:(subset+1)*sub_len].cuda()
            G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

            l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape).cuda())
            sinogram_eval = (sub_mul_factor.reshape(-1,1) * l.reshape(-1,1) + sub_pet_noise.reshape(-1,1)).detach().cpu()
            tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
            # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
            backproj = subset_radons[subset].backward((sub_mul_factor.reshape(-1,1)*tmp.float().cuda()).reshape(sub_mul_factor.shape))
            sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(-1,1)
            # sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
            # print(torch.sum(sub_sinogram - sinogram_eval))

        sub_pet_es.append(sub_pet_eval.detach().cpu().numpy().reshape(pet_gt1.shape))

    # img = sub_pet_eval.detach().cpu().numpy().reshape(pet_gt1.shape)

    return sub_pet_es[-1],sub_pet_es[-2],sub_pet_es[-3]

cpu_worker_num = 4
qbar1 = trange(20400//cpu_worker_num, desc="Progress 1", leave=False) #67383
qbar2 = trange(20400//cpu_worker_num, desc="Progress 2", leave=False)
qbar3 = trange(20400//cpu_worker_num, desc="Progress 3", leave=False)
qbar4 = trange(20400//cpu_worker_num, desc="Progress 4", leave=False)
# qbar5 = trange(20400//cpu_worker_num, desc="Progress 5", leave=False) #67383
# qbar6 = trange(20400//cpu_worker_num, desc="Progress 6", leave=False)
# qbar7 = trange(20400//cpu_worker_num, desc="Progress 7", leave=False)
# qbar8 = trange(20400//cpu_worker_num, desc="Progress 8", leave=False)
# qbars = [qbar1,qbar2,qbar3,qbar4,qbar5,qbar6,qbar7,qbar8]
qbars = [qbar1,qbar2,qbar3,qbar4]
def process_EM(args):
    curr_proc = multiprocessing.current_process() 
    id = curr_proc._identity[0]
    x = args

    kemresults = np.zeros((5,128,128))
    # pet_gt1 = normalize(x)
    mr = x[1]
    # mr = normalize(mr)
    pet_gt1 = x[0]
    # pet_gt1 = normalize(pet_gt1)
    
    #old
    # pet_gt_t = x[0]
    # np.maximum(pet_gt_t, 0, out=pet_gt_t)
    # xmax = np.amax(pet_gt_t,axis=(-1,-2),keepdims=True)
    # xmax_tmp = np.ones_like(xmax)
    # xmax_tmp[xmax<0.1]=0.01
    # xmax_tmp[xmax<1]=0.1
    # xmax_tmp[xmax>10]=10
    # xmax_tmp[xmax>100]=100
    # xmax_tmp[xmax>1000]=1000
    # xmax_tmp[xmax>10000]=10000
    # xmax_tmp[xmax>1000000]=1000000
    # xmax_tmp[xmax>10000000]=10000000
    # xmax_tmp[xmax>100000000]=100000000
    # xmax_tmp[xmax>1000000000]=1000000000
    # xmax_tmp[xmax>10000000000]=10000000000
    # xmax = xmax_tmp
    # pet_gt1 = pet_gt_t/xmax
    # pet_gt1 = x

    # 投影矩阵
    radon = ParallelBeam(128,np.linspace(0,np.pi,128,False))
    # 从PET图像中仿真正弦图（sinogram）
    sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=20e4)

    sinogram = torch.Tensor(sinogram)
    mul_factor = torch.Tensor(mul_factor)
    pet_noise = torch.Tensor(pet_noise)

    #gassian kernel
    kernel_size = 128*128
    sigma = 100 #0.05 100
    # gaussian_matrix = gaussian_kernel(mr,kernel_size, sigma,J=2)

    #wrong kernel
    # gaussian_matrix = gaussian_kernel(pet_gt1,kernel_size, sigma,J=2)

    #RKEM
    # kemresult1,kemresult2,kemresult3 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,K=None,method_index='KEM',prefix='KEM',total_iter=20)
    # gaussian_matrix = gaussian_kernel2(kemresult1,mr,kernel_size, sigma,J=2)
    gaussian_matrix = None

    # kemresult1 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=10)
    # kemresult2 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=20)
    kemresult1,kemresult2,kemresult3,kemresult4,kemresult5 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,pet_gt1,K=gaussian_matrix,method_index='KEM',prefix='KEM',radon=radon,total_iter=50)
    
    #OSEM
    # kemresult1,kemresult2,kemresult3 = OSEM(sinogram,mul_factor,pet_noise,pet_gt1,8,4,128)

    kemresults[0] = kemresult1
    kemresults[1] = kemresult2
    kemresults[2] = kemresult3
    kemresults[3] = kemresult4
    kemresults[4] = kemresult5

    psnr1 = compare_psnr(pet_gt1,kemresult1,data_range=np.max(pet_gt1))
    ssim1 = compare_ssim(kemresult1,pet_gt1,data_range=np.max(pet_gt1))

    qbars[id % cpu_worker_num].update(1)
    qbars[id % cpu_worker_num].set_postfix(psnr=psnr1,ssim=ssim1) 

    return kemresults

if __name__=="__main__":


    # path = r'F:\brainweb\pet_vol_slice_144by144_nt.npy'
    test_id = np.load('./test_list1.npy')
    pet_data = np.load('/mnt/c/brainweb/pet_vol_slice_144by144_nt.npy')[test_id]
    mr_data = np.load('/mnt/c/brainweb/pet_vol_slice_144by144_nt_mr.npy')[test_id]

    # pet_data = np.load('/mnt/d/HECKTOR-test.npy')
    # mr_data = np.load('/mnt/d/HECKTOR-test-ct.npy')

    # data1 = read_data("pet_new.mat")

    # path = 'test1.npy'

    # data = np.load(path)

    print('GT generating',len(pet_data))
    inputs = []

    for i in range(0,len(pet_data)):
        # plt.subplot(1,2,1)
        pet_gt,mr = np.array(pet_data[i]),np.array(mr_data[i])
        idata = pad_and_center_image(np.stack([pet_gt,mr], axis=0),(128,128))
        # pet_gt1 = pad_and_center_image(data[i],(128,128))
        # plt.imshow(pet_gt1)
        # plt.subplot(1,2,2)
        # pet_gt1 = data[i][32:160,32:160]
        # plt.imshow(pet_gt1)



        if np.sum(pet_gt) == 0:
            # print('pass')
            continue
        # plt.show()
        inputs.append(idata)

    
    print('inputLen',len(inputs))
    print('parllen generating')
    with Pool(cpu_worker_num) as p:
        outputs = p.map(process_EM, inputs)
    print(len(inputs),len(outputs))
    np.savez('./datasetB5N-EM-test.npz',kem=outputs)

