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
cpu_worker_num = 8
qbar1 = trange(33459//cpu_worker_num, desc="Progress 1", leave=False) #67383
qbar2 = trange(33459//cpu_worker_num, desc="Progress 2", leave=False)
qbar3 = trange(33459//cpu_worker_num, desc="Progress 3", leave=False)
qbar4 = trange(33459//cpu_worker_num, desc="Progress 4", leave=False)
qbar5 = trange(33459//cpu_worker_num, desc="Progress 5", leave=False)
qbar6 = trange(33459//cpu_worker_num, desc="Progress 6", leave=False)
qbar7 = trange(33459//cpu_worker_num, desc="Progress 7", leave=False)
qbar8 = trange(33459//cpu_worker_num, desc="Progress 8", leave=False)
qbars = [qbar1,qbar2,qbar3,qbar4,qbar5,qbar6,qbar7,qbar8]
from scipy.sparse import coo_matrix
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
                row = i // 160 + j
                col = i % 160 + k
                # 确保索引在矩阵范围内
                if 0 <= row < 160 and 0 <= col < 160:
                    dist = np.abs(j) + np.abs(k)
                    if dist <= J:
                        index = row * 160 + col
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

    # 将X展平为一维数组
    x_hat = X.flatten()
    
    # 计算所有点对的距离矩阵
    rows = np.arange(kernel_size)
    cols = np.arange(kernel_size)
    rows_grid, cols_grid = np.meshgrid(rows, cols)
    dist_matrix = np.abs(rows_grid - cols_grid) % 128 + np.abs((rows_grid - cols_grid) // 128)
    
    # 创建一个与距离矩阵同样大小的布尔矩阵，标记距离小于J的位置
    mask = dist_matrix < J
    
    # 只计算距离小于J的元素对应的核矩阵值
    x_hat_diff = x_hat[rows_grid[mask]] - x_hat[cols_grid[mask]]
    kernel_values = np.exp(-(x_hat_diff ** 2) / (2 * sigma ** 2))
    
    # 初始化核矩阵
    kernel = np.zeros((kernel_size, kernel_size))
    
    # 将计算得到的核矩阵值赋给对应的位置
    kernel[mask] = kernel_values

    # 由于核矩阵是对称的，所以将上三角复制到下三角
    kernel = kernel + kernel.T - np.diag(np.diag(kernel))
    
    return kernel
    # kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2) / (2 * sigma**2)), (kernel_size, kernel_size))
    # normal = kernel / np.sum(kernel)

    # print(X.shape, x_hat.shape)
    # kernel = np.zeros((kernel_size,kernel_size))
    # for i in range(kernel_size):
    #     # for j in range(kernel_size):
    #     #     dist = np.abs(i%128-j%128) + np.abs(i/128 - j/128)
    #     #     if (dist < J):
    #     #         kernel[i][j] = np.exp(-(x_hat[i] - x_hat[j])**2 / 2*sigma**2)
    #     if (i+1 < kernel_size):
    #         kernel[i][i+1] = np.exp(-(x_hat[i] - x_hat[i+1])**2 / 2*sigma**2)
    #     if (i+128 < kernel_size):
    #         kernel[i+1][i] = np.exp(-(x_hat[i] - x_hat[i+128])**2 / 2*sigma**2)
    #     if (i-1 >= 0):
    #         kernel[i][i-1] = np.exp(-(x_hat[i] - x_hat[i-1])**2 / 2*sigma**2)
    #     if (i-128 >= 0):
    #         kernel[i-1][i] = np.exp(-(x_hat[i] - x_hat[i-128])**2 / 2*sigma**2)

    #     if (i+128+1 < kernel_size and i+1 < kernel_size):
    #         kernel[i+1][i+1] = np.exp(-(x_hat[i] - x_hat[i+128+1])**2 / 2*sigma**2)
    #     if (i+128-1 < kernel_size and i-1 >= 0):
    #         kernel[i+1][i-1] = np.exp(-(x_hat[i] - x_hat[i+128-1])**2 / 2*sigma**2)
    #     if (i-128 >= 0 and i+1 < kernel_size):
    #         kernel[i-1][i+1] = np.exp(-(x_hat[i] - x_hat[i-128+1])**2 / 2*sigma**2)
    #     if (i-128-1 >= 0 and i-1 >= 0):
    #         kernel[i-1][i-1] = np.exp(-(x_hat[i] - x_hat[i-128-1])**2 / 2*sigma**2)

    x_hat = X.reshape(-1,1)
    # 计算距离矩阵
    rows, cols = np.indices((kernel_size, kernel_size))
    dist_matrix = np.abs(rows % 160 - cols % 160) + np.abs(rows // 160 - cols // 160)

    # 初始化kernel矩阵
    kernel = np.zeros((kernel_size, kernel_size))
    # kernel =scipy.sparse.identity(np.prod(X.shape), format='coo')

    # 计算高斯核函数的值
    gaussian_matrix = np.exp(-(x_hat - x_hat.T)**2 / (2 * sigma**2))

    # 根据距离矩阵和条件设置kernel的值
    kernel[dist_matrix < J] = gaussian_matrix[dist_matrix < J]

    # X = torch.Tensor(X).cuda()  # 示例数据，移动到GPU
    # x_hat = X.view(-1, 1)

    # # 计算距离矩阵
    # rows, cols = torch.meshgrid(torch.arange(kernel_size, dtype=torch.float32).cuda(), 
    #                             torch.arange(kernel_size, dtype=torch.float32).cuda(),indexing="ij")
    # dist_matrix = torch.abs(rows % 128 - cols % 128) + torch.abs(rows // 128 - cols // 128)

    # # 初始化kernel矩阵
    # kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32).cuda()

    # # 计算高斯核函数的值
    # gaussian_matrix = torch.exp(-(x_hat - x_hat.t())**2 / (2 * sigma**2))

    # # 根据距离矩阵和条件设置kernel的值
    # kernel[dist_matrix < J] = gaussian_matrix[dist_matrix < J]

    return kernel


def process_EM(args):
    curr_proc = multiprocessing.current_process() 
    id = curr_proc._identity[0]
    x = args

    kemresults = np.zeros((3,160,160))
    # pet_gt1 = normalize(x)*random.uniform(1,8)
    # pet_gt1 = normalize(x[0])
    pet_gt1 = np.clip(x[0]/30000,0,1)
    ct = x[1]
    # 投影矩阵
    radon = ParallelBeam(160,np.linspace(0,180,160,False))
    # 从PET图像中仿真正弦图（sinogram）
    sinogram,mul_factor,pet_noise = PET2Sinogram(pet_gt1,radon,count=20e4)

    sinogram = torch.Tensor(sinogram)
    mul_factor = torch.Tensor(mul_factor)
    pet_noise = torch.Tensor(pet_noise)

    kernel_size = 160*160
    sigma = 100 #100
    # gaussian_matrix = gaussian_kernel2(ct,ct,kernel_size, sigma,J=2)
    # kemresult1 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=10)
    # kemresult2 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=20)
    # gaussian_matrix =scipy.sparse.identity(np.prod(pet_gt1.shape), format='coo')
    kemresult1,kemresult2,kemresult3 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,K=None,method_index='KEM',prefix='KEM',total_iter=20)
    gaussian_matrix = gaussian_kernel2(kemresult1,ct,kernel_size, sigma,J=2)


    kemresult1,kemresult2,kemresult3 = MLEMorKEM(sinogram,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,K=gaussian_matrix,method_index='KEM',prefix='KEM',total_iter=30)

    kemresults[0] = kemresult1
    kemresults[1] = kemresult2
    kemresults[2] = kemresult3


    qbars[id % cpu_worker_num].update(1)

    return kemresults

if __name__=="__main__":
    # path = './totalbodyDataset/rawdata.npy'
    path = './ctpetraw.npz'

    rdata = np.load(path)

    data = np.concatenate((np.expand_dims(rdata['PET'],axis=1),np.expand_dims(rdata['CT'],axis=1)),axis=1)

    print(data.shape)
    print('GT generating')

    n_examples = len(data)

    n_train = int(n_examples * 0.8) 
    inputs = []

    for i in range(n_train,len(data)):
        # plt.subplot(1,2,1)
        pet_gt1 = data[i]
        # plt.imshow(pet_gt1)
        # plt.subplot(1,2,2)
        # pet_gt1 = data[i][32:160,32:160]
        # plt.imshow(pet_gt1)



        if np.sum(pet_gt1) == 0:
            # print('pass')
            continue
        # plt.show()
        inputs.append(pet_gt1)

    
    print('inputLen',len(inputs))
    print('parllen generating')
    with Pool(cpu_worker_num) as p:
        outputs = p.map(process_EM, inputs)
    print(len(inputs),len(outputs))
    np.savez('./datasetK3-PETCT-RKEM-192T-Ga-Test.npz',kem=outputs)

