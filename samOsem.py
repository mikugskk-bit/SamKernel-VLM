from argparse import Namespace
import torch
import numpy as np
import cv2
from predictor_sammed import SammedPredictor
from build_sam import sam_model_registry
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio as PSNR
from torchmetrics import MeanSquaredError

mean_squared_error = MeanSquaredError().to('cuda')

from torch_radon import ParallelBeam
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



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        # x = self.block_5_2_left(self.block_5_1_left(self.pool_3(ds3)))

        # x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(x), ds3], 1)))
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)
        # return x
        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim , action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class PolicyBoxNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyBoxNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim , action_dim)
        self.fc3 = torch.nn.Linear(hidden_dim , action_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1),F.softmax(self.fc3(x), dim=1)



class SamAgentOsem(nn.Module):
    def __init__(self,seve_all = False,use_box = True, use_point = True, unfreeze = False):
        super().__init__()
        args = Namespace()

        num_subsets = 16
        self.num_subsets = num_subsets
        self.total_iter = 8

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/mnt/c/sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        self.unet = Unet(in_channels=self.total_iter+1, n_cls=1, n_filters=16)

        self.act1 = PolicyNet(256,128,256)
        self.act2 = PolicyNet(256,128,256)
        self.box1 = PolicyBoxNet(64,128,256)
        self.box2 = PolicyBoxNet(64,128,256)

        self.lin3 = nn.Linear(64,128)
        self.boxhdim = nn.Linear(128,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.statePool = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.statePool_p = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv_p = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(self.total_iter , self.total_iter , kernel_size=3, stride=1, padding=1)
        
        self.count = 0



        det_count = 160
        all_angles = np.linspace(0,np.pi,160,False)
        self.subset_radons = []
        sub_len = int(len(all_angles) / num_subsets)
        self.sub_len = sub_len
        for i in range(num_subsets): 
            angles = all_angles[i*sub_len:(i+1)*sub_len]
            sub_radon = ParallelBeam(det_count,angles)
            self.subset_radons.append(sub_radon)


    def forward(self, inp,gt = None,save_all = False):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.unfreeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True

        sinogram = inp[:,0]
        mul_factor = inp[:,1]
        pet_noise = inp[:,2]

        num_subsets = self.num_subsets
        total_iter = self.total_iter
        pet_gt1 = torch.zeros((inp.shape[0],1,160,160)).cuda()
        sinogram = sinogram.reshape(pet_gt1.shape)
        mul_factor = mul_factor.reshape(pet_gt1.shape)
        pet_noise = pet_noise.reshape(pet_gt1.shape)

        subset_radons = self.subset_radons
        sub_len = self.sub_len

        # 迭代重建
        sub_pet_eval = torch.ones_like(pet_gt1).reshape(inp.shape[0],1,-1,1)


        mask_list = []
        points_list = []
        boxs_list = []
        xo_list = []

        for i in range(total_iter):
            for subset in range(num_subsets):
                sub_sinogram = sinogram[:,:,subset*sub_len:(subset+1)*sub_len].reshape(inp.shape[0],1,-1,1)
                sub_mul_factor = mul_factor[:,:,subset*sub_len:(subset+1)*sub_len]
                sub_pet_noise = pet_noise[:,:,subset*sub_len:(subset+1)*sub_len]
                G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

                l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape))
                sinogram_eval = (sub_mul_factor.reshape(inp.shape[0],1,-1,1) * l.reshape(inp.shape[0],1,-1,1) + sub_pet_noise.reshape(inp.shape[0],1,-1,1))
                tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
                # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
                backproj = subset_radons[subset].backward((sub_mul_factor.reshape(inp.shape[0],1,-1,1)*tmp).reshape(sub_mul_factor.shape))
                sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(inp.shape[0],1,-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(inp.shape[0],1,-1,1)
                # sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
                # print(torch.sum(sub_sinogram - sinogram_eval))



            x = sub_pet_eval.reshape(pet_gt1.shape)

            xo = x
            xo_list.append(xo)

            x = self.upnear(x)
            xinput = x
            ##Single

            pl = torch.Tensor([1,1,1,1,1,1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
            # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

            mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

            image_embeddings = self.model.image_encoder(x.repeat(1,3,1,1)*256)

            # Embed prompts
            multimask_output = True

            # last_x = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
            # last_y = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
            # xi = x[:,2-i:3-i,:,:]
            xi = x

            if self.use_point:
                xstate2 = self.stateConv_p(self.statePool_p(xi)).reshape(xi.shape[0],256)
                px = self.act1(xstate2)
                py = self.act2(xstate2)
                s_x = torch.argmax(px,dim=1)
                s_y = torch.argmax(py,dim=1)
                # last_x = s_x
                # last_y = s_y
                
                point = torch.cat([s_x.unsqueeze(1).unsqueeze(1),s_y.unsqueeze(1).unsqueeze(1)],dim=2)

                points_list.append(point)
                points = (torch.cat(points_list,dim=1), pl[:,:i+1])
            else:
                points = None

            if self.use_box:
                #old
                # xstate1 = self.stateConv(self.statePool(xi)).reshape(xi.shape[0],64)
                # bx1,by1 = self.box1(xstate1)
                # bx2,by2 = self.box2(xstate1)
                # bx1 = torch.argmax(bx1,dim=1).unsqueeze(1)
                # bx2 = torch.argmax(bx2,dim=1).unsqueeze(1)
                # by1 = torch.argmax(by1,dim=1).unsqueeze(1)
                # by2 = torch.argmax(by2,dim=1).unsqueeze(1)
                #new
                xstate1 = self.stateConv(self.statePool(xi)) # B,1,16,16
                cx = torch.argmax(torch.sum(xstate1,dim=-1),dim=-1) * 10 /160*256 # -1 -2
                cy = torch.argmax(torch.sum(xstate1,dim=-2),dim=-1) * 10 /160*256 # -2 -1
                nozerox = torch.count_nonzero(xstate1>0.5,dim=-1)
                nozeroy = torch.count_nonzero(xstate1>0.5,dim=-2)
                squarex = nozerox.max(dim=-1)[0]*10# 面积等价 x y
                squarey = nozeroy.max(dim=-1)[0]*10# 面积等价 y x
                # print(squarex,squarey)
                bx1 = cx + squarex *0.5
                bx2 = cx - squarex *0.5
                by1 = cy + squarey *0.5
                by2 = cy - squarey *0.5
                

                pre_boxes = torch.cat([bx1,by1,bx2,by2],dim=1)

            else:
                pre_boxes = None


            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=pre_boxes,
                masks=mask_input,
            )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.model.postprocess_masks(
                low_res_masks,
                input_size=x.shape[-2:],
                original_size=256,
            )
            masks = torch.sigmoid(masks)
            # masks = (masks > 0.5).float()

            mask_list.append(masks)

            boxs_list.append(pre_boxes)

            sub_pet_eval = sub_pet_eval + self.sampledown(masks).reshape(inp.shape[0],1,-1,1)


        masks = torch.cat(mask_list, dim=1)


        y = masks
        y = self.downconv(self.sampledown(y))

        x = torch.cat([xo,y],dim=1)
        x = self.unet(x)

        if save_all:
            np.savez('./promptResPoint-8E/batch-'+str(self.count)+'.npz',input=xinput.detach().cpu().numpy(),
                    sam_mask = masks.detach().cpu().numpy(),
                    point=torch.cat(points_list, dim=1).detach().cpu().numpy(),
                    box = torch.cat(boxs_list, dim=1).cpu().numpy(),
                    imask = mask_input.detach().cpu().numpy(),
                    sam_out = y.detach().cpu().numpy(),
                    output = x.detach().cpu().numpy(),
                    xo = torch.cat(xo_list,dim=1).detach().cpu().numpy(),
                    gt = gt.detach().cpu().numpy())
            self.count = self.count+1
        return x,y
    
class SamAgentEmBrO(nn.Module):
    def __init__(self,seve_all = False,use_box = True, use_point = True, unfreeze = False):
        super().__init__()
        args = Namespace()

        num_subsets = 1
        self.num_subsets = num_subsets
        self.total_iter = 30

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/mnt/c/sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        self.unet = Unet(in_channels=self.total_iter//10+1, n_cls=1, n_filters=16)

        self.act1 = PolicyNet(256,128,256)
        self.act2 = PolicyNet(256,128,256)
        self.box1 = PolicyBoxNet(64,128,256)
        self.box2 = PolicyBoxNet(64,128,256)

        self.lin3 = nn.Linear(64,128)
        self.boxhdim = nn.Linear(128,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.statePool = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.statePool_p = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv_p = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((128,128))

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(self.total_iter//10 , self.total_iter//10 , kernel_size=3, stride=1, padding=1)
        
        self.count = 0



        det_count = 128
        all_angles = np.linspace(0,np.pi,128,False)
        self.subset_radons = []
        sub_len = int(len(all_angles) / num_subsets)
        self.sub_len = sub_len
        for i in range(num_subsets): 
            angles = all_angles[i*sub_len:(i+1)*sub_len]
            sub_radon = ParallelBeam(det_count,angles)
            self.subset_radons.append(sub_radon)


    def forward(self, inp,gt = None,save_all = False):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.unfreeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True

        sinogram = inp[:,0]
        mul_factor = inp[:,1]
        pet_noise = inp[:,2]

        num_subsets = self.num_subsets
        total_iter = self.total_iter
        pet_gt1 = torch.zeros((inp.shape[0],1,128,128)).cuda()
        sinogram = sinogram.reshape(pet_gt1.shape)
        mul_factor = mul_factor.reshape(pet_gt1.shape)
        pet_noise = pet_noise.reshape(pet_gt1.shape)

        subset_radons = self.subset_radons
        sub_len = self.sub_len

        # 迭代重建
        sub_pet_eval = torch.ones_like(pet_gt1).reshape(inp.shape[0],1,-1,1)


        mask_list = []
        points_list = []
        boxs_list = []

        ic = 0
        for i in range(total_iter):
            for subset in range(num_subsets):
                sub_sinogram = sinogram[:,:,subset*sub_len:(subset+1)*sub_len].reshape(inp.shape[0],1,-1,1)
                sub_mul_factor = mul_factor[:,:,subset*sub_len:(subset+1)*sub_len]
                sub_pet_noise = pet_noise[:,:,subset*sub_len:(subset+1)*sub_len]
                G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

                l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape))
                sinogram_eval = (sub_mul_factor.reshape(inp.shape[0],1,-1,1) * l.reshape(inp.shape[0],1,-1,1) + sub_pet_noise.reshape(inp.shape[0],1,-1,1))
                tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
                # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
                backproj = subset_radons[subset].backward((sub_mul_factor.reshape(inp.shape[0],1,-1,1)*tmp).reshape(sub_mul_factor.shape))
                sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(inp.shape[0],1,-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(inp.shape[0],1,-1,1)
                # sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
                # print(torch.sum(sub_sinogram - sinogram_eval))


            if i == 0 or i == 10 or i == 20: #9 19 29 ;0 10 20 
                x = sub_pet_eval.reshape(pet_gt1.shape)

                xo = x

                x = self.upnear(x)
                xinput = x
                ##Single

                pl = torch.Tensor([1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
                # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

                mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

                image_embeddings = self.model.image_encoder(x.repeat(1,3,1,1)*256)

                # Embed prompts
                multimask_output = False

                # last_x = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
                # last_y = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
                # xi = x[:,2-i:3-i,:,:]
                xi = x

                if self.use_point:
                    xstate2 = self.stateConv_p(self.statePool_p(xi)).reshape(xi.shape[0],256)
                    px = self.act1(xstate2)
                    py = self.act2(xstate2)
                    s_x = torch.argmax(px,dim=1)
                    s_y = torch.argmax(py,dim=1)
                    # last_x = s_x
                    # last_y = s_y
                    
                    point = torch.cat([s_x.unsqueeze(1).unsqueeze(1),s_y.unsqueeze(1).unsqueeze(1)],dim=2)

                    points_list.append(point)
                    points = (torch.cat(points_list,dim=1), pl[:,:ic+1])
                    ic = ic + 1
                else:
                    points = None

                if self.use_box:
                    #old
                    # xstate1 = self.stateConv(self.statePool(xi)).reshape(xi.shape[0],64)
                    # bx1,by1 = self.box1(xstate1)
                    # bx2,by2 = self.box2(xstate1)
                    # bx1 = torch.argmax(bx1,dim=1).unsqueeze(1)
                    # bx2 = torch.argmax(bx2,dim=1).unsqueeze(1)
                    # by1 = torch.argmax(by1,dim=1).unsqueeze(1)
                    # by2 = torch.argmax(by2,dim=1).unsqueeze(1)
                    #new
                    xstate1 = self.stateConv(self.statePool(xi)) # B,1,16,16
                    cx = torch.argmax(torch.sum(xstate1,dim=-1),dim=-1) * 10 /160*256 # -1 -2
                    cy = torch.argmax(torch.sum(xstate1,dim=-2),dim=-1) * 10 /160*256 # -2 -1
                    nozerox = torch.count_nonzero(xstate1>0.5,dim=-1)
                    nozeroy = torch.count_nonzero(xstate1>0.5,dim=-2)
                    squarex = nozerox.max(dim=-1)[0]*10# 面积等价 x y
                    squarey = nozeroy.max(dim=-1)[0]*10# 面积等价 y x
                    # print(squarex,squarey)
                    bx1 = cx + squarex *0.5
                    bx2 = cx - squarex *0.5
                    by1 = cy + squarey *0.5
                    by2 = cy - squarey *0.5
                    

                    pre_boxes = torch.cat([bx1,by1,bx2,by2],dim=1)

                else:
                    pre_boxes = None


                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=pre_boxes,
                    masks=mask_input,
                )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                masks = self.model.postprocess_masks(
                    low_res_masks,
                    input_size=x.shape[-2:],
                    original_size=256,
                )
                masks = torch.sigmoid(masks)
                # masks = (masks > 0.5).float()

                mask_list.append(masks)

                boxs_list.append(pre_boxes)

                sub_pet_eval = sub_pet_eval + self.sampledown(masks).reshape(inp.shape[0],1,-1,1)

        masks = torch.cat(mask_list, dim=1)


        y = masks
        y = self.downconv(self.sampledown(y))

        x = torch.cat([xo,y],dim=1)
        x = self.unet(x)

        # if save_all:
        #     np.savez('./promptResPoint-EM-i1/batch-'+str(self.count)+'.npz',input=xinput.detach().cpu().numpy(),
        #             sam_mask = masks.detach().cpu().numpy(),
        #             point=torch.cat(points_list, dim=1).detach().cpu().numpy(),
        #             box = torch.cat(boxs_list, dim=1).cpu().numpy(),
        #             imask = mask_input.detach().cpu().numpy(),
        #             sam_out = y.detach().cpu().numpy(),
        #             output = x.detach().cpu().numpy(),
        #             xo = xo.detach().cpu().numpy(),
        #             gt = gt.detach().cpu().numpy())
        #     self.count = self.count+1
        return x,y


class SamAgentOsemNew(nn.Module):
    def __init__(self,seve_all = False,use_box = True, use_point = True, unfreeze = True):
        super().__init__()
        args = Namespace()

        num_subsets = 16
        self.num_subsets = num_subsets
        self.total_iter = 8

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/mnt/c/sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        self.act1 = PolicyNet(256,128,256)
        self.act2 = PolicyNet(256,128,256)
        self.box1 = PolicyBoxNet(64,128,256)
        self.box2 = PolicyBoxNet(64,128,256)

        self.lin3 = nn.Linear(64,128)
        self.boxhdim = nn.Linear(128,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.statePool = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.statePool_p = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv_p = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(self.total_iter , self.total_iter , kernel_size=3, stride=1, padding=1)
        
        self.count = 0



        det_count = 160
        all_angles = np.linspace(0,np.pi,160,False)
        self.subset_radons = []
        sub_len = int(len(all_angles) / num_subsets)
        self.sub_len = sub_len
        for i in range(num_subsets): 
            angles = all_angles[i*sub_len:(i+1)*sub_len]
            sub_radon = ParallelBeam(det_count,angles)
            self.subset_radons.append(sub_radon)


    def forward(self, inp,gt = None,save_all = False):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.unfreeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True

        sinogram = inp[:,0]
        mul_factor = inp[:,1]
        pet_noise = inp[:,2]

        num_subsets = self.num_subsets
        total_iter = self.total_iter
        pet_gt1 = torch.zeros((inp.shape[0],1,160,160)).cuda()
        sinogram = sinogram.reshape(pet_gt1.shape)
        mul_factor = mul_factor.reshape(pet_gt1.shape)
        pet_noise = pet_noise.reshape(pet_gt1.shape)

        subset_radons = self.subset_radons
        sub_len = self.sub_len

        # 迭代重建
        sub_pet_eval = torch.ones_like(pet_gt1).reshape(inp.shape[0],1,-1,1)


        mask_list = []
        points_list = []
        boxs_list = []
        xo_list = []

        for i in range(total_iter):
            for subset in range(num_subsets):
                sub_sinogram = sinogram[:,:,subset*sub_len:(subset+1)*sub_len].reshape(inp.shape[0],1,-1,1)
                sub_mul_factor = mul_factor[:,:,subset*sub_len:(subset+1)*sub_len]
                sub_pet_noise = pet_noise[:,:,subset*sub_len:(subset+1)*sub_len]
                G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

                l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape))
                sinogram_eval = (sub_mul_factor.reshape(inp.shape[0],1,-1,1) * l.reshape(inp.shape[0],1,-1,1) + sub_pet_noise.reshape(inp.shape[0],1,-1,1))
                tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
                # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
                backproj = subset_radons[subset].backward((sub_mul_factor.reshape(inp.shape[0],1,-1,1)*tmp).reshape(sub_mul_factor.shape))
                sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(inp.shape[0],1,-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(inp.shape[0],1,-1,1)
                # sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
                # print(torch.sum(sub_sinogram - sinogram_eval))



            x = sub_pet_eval.reshape(pet_gt1.shape)

            xo = x
            xo_list.append(xo)

            x = self.upnear(x)
            xinput = x
            ##Single

            pl = torch.Tensor([1,1,1,1,1,1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
            # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

            mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

            image_embeddings = self.model.image_encoder(x.repeat(1,3,1,1)*256)

            # Embed prompts
            multimask_output = False

            # last_x = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
            # last_y = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
            # xi = x[:,2-i:3-i,:,:]
            xi = x

            if self.use_point:
                xstate2 = self.stateConv_p(self.statePool_p(xi)).reshape(xi.shape[0],256)
                px = self.act1(xstate2)
                py = self.act2(xstate2)
                s_x = torch.argmax(px,dim=1)
                s_y = torch.argmax(py,dim=1)
                # last_x = s_x
                # last_y = s_y
                
                point = torch.cat([s_x.unsqueeze(1).unsqueeze(1),s_y.unsqueeze(1).unsqueeze(1)],dim=2)

                points_list.append(point)
                points = (torch.cat(points_list[-1:],dim=1), pl[:,i:i+1])
            else:
                points = None

            if self.use_box:
                #old
                # xstate1 = self.stateConv(self.statePool(xi)).reshape(xi.shape[0],64)
                # bx1,by1 = self.box1(xstate1)
                # bx2,by2 = self.box2(xstate1)
                # bx1 = torch.argmax(bx1,dim=1).unsqueeze(1)
                # bx2 = torch.argmax(bx2,dim=1).unsqueeze(1)
                # by1 = torch.argmax(by1,dim=1).unsqueeze(1)
                # by2 = torch.argmax(by2,dim=1).unsqueeze(1)
                #new
                xstate1 = self.stateConv(self.statePool(xi)) # B,1,16,16
                cx = torch.argmax(torch.sum(xstate1,dim=-1),dim=-1) * 10 /160*256 # -1 -2
                cy = torch.argmax(torch.sum(xstate1,dim=-2),dim=-1) * 10 /160*256 # -2 -1
                nozerox = torch.count_nonzero(xstate1>0.5,dim=-1)
                nozeroy = torch.count_nonzero(xstate1>0.5,dim=-2)
                squarex = nozerox.max(dim=-1)[0]*10# 面积等价 x y
                squarey = nozeroy.max(dim=-1)[0]*10# 面积等价 y x
                # print(squarex,squarey)
                bx1 = cx + squarex *0.5
                bx2 = cx - squarex *0.5
                by1 = cy + squarey *0.5
                by2 = cy - squarey *0.5
                

                pre_boxes = torch.cat([bx1,by1,bx2,by2],dim=1)

            else:
                pre_boxes = None


            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=pre_boxes,
                masks=mask_input,
            )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.model.postprocess_masks(
                low_res_masks,
                input_size=x.shape[-2:],
                original_size=256,
            )
            masks = torch.sigmoid(masks)
            # masks = (masks > 0.5).float()

            mask_list.append(masks)

            boxs_list.append(pre_boxes)

            sub_pet_eval = sub_pet_eval + self.sampledown(masks).reshape(inp.shape[0],1,-1,1)

        masks = torch.cat(mask_list, dim=1)


        y = masks
        y = self.downconv(self.sampledown(y))

        x = sub_pet_eval.reshape(pet_gt1.shape)
        x = self.sampledown(x)

        if save_all:
            np.savez('./promptResPoint-8NU/batch-'+str(self.count)+'.npz',input=xinput.detach().cpu().numpy(),
                    sam_mask = masks.detach().cpu().numpy(),
                    point=torch.cat(points_list, dim=1).detach().cpu().numpy(),
                    box = torch.cat(boxs_list, dim=1).cpu().numpy(),
                    imask = mask_input.detach().cpu().numpy(),
                    sam_out = y.detach().cpu().numpy(),
                    output = x.detach().cpu().numpy(),
                    xo = torch.cat(xo_list,dim=1).detach().cpu().numpy(),
                    gt = gt.detach().cpu().numpy())
            self.count = self.count+1
        return x,y



class SamAgentEm(nn.Module):
    def __init__(self,seve_all = False,use_box = True, use_point = True, unfreeze = False):
        super().__init__()
        args = Namespace()

        num_subsets = 1
        self.num_subsets = num_subsets
        self.total_iter = 30

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/mnt/c/sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        self.unet = Unet(in_channels=self.total_iter//10+1, n_cls=1, n_filters=16)

        self.act1 = PolicyNet(256,128,256)
        self.act2 = PolicyNet(256,128,256)
        self.box1 = PolicyBoxNet(64,128,256)
        self.box2 = PolicyBoxNet(64,128,256)

        self.lin3 = nn.Linear(64,128)
        self.boxhdim = nn.Linear(128,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.statePool = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.statePool_p = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv_p = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(self.total_iter//10 , self.total_iter//10 , kernel_size=3, stride=1, padding=1)
        
        self.count = 0

        #loss test
        self.losses = torch.zeros(30)



        det_count = 160
        all_angles = np.linspace(0,np.pi,160,False)
        self.subset_radons = []
        sub_len = int(len(all_angles) / num_subsets)
        self.sub_len = sub_len
        for i in range(num_subsets): 
            angles = all_angles[i*sub_len:(i+1)*sub_len]
            sub_radon = ParallelBeam(det_count,angles)
            self.subset_radons.append(sub_radon)


    def forward(self, inp,gt = None,save_all = False):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.unfreeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True

        sinogram = inp[:,0]
        mul_factor = inp[:,1]
        pet_noise = inp[:,2]

        num_subsets = self.num_subsets
        total_iter = self.total_iter
        pet_gt1 = torch.zeros((inp.shape[0],1,160,160)).cuda()
        sinogram = sinogram.reshape(pet_gt1.shape)
        mul_factor = mul_factor.reshape(pet_gt1.shape)
        pet_noise = pet_noise.reshape(pet_gt1.shape)

        subset_radons = self.subset_radons
        sub_len = self.sub_len

        # 迭代重建
        sub_pet_eval = torch.ones_like(pet_gt1).reshape(inp.shape[0],1,-1,1)


        mask_list = []
        points_list = []
        boxs_list = []

        ic = 0
        for i in range(total_iter):
            for subset in range(num_subsets):
                sub_sinogram = sinogram[:,:,subset*sub_len:(subset+1)*sub_len].reshape(inp.shape[0],1,-1,1)
                sub_mul_factor = mul_factor[:,:,subset*sub_len:(subset+1)*sub_len]
                sub_pet_noise = pet_noise[:,:,subset*sub_len:(subset+1)*sub_len]
                G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

                l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape))
                sinogram_eval = (sub_mul_factor.reshape(inp.shape[0],1,-1,1) * l.reshape(inp.shape[0],1,-1,1) + sub_pet_noise.reshape(inp.shape[0],1,-1,1))
                tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
                # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
                backproj = subset_radons[subset].backward((sub_mul_factor.reshape(inp.shape[0],1,-1,1)*tmp).reshape(sub_mul_factor.shape))
                sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(inp.shape[0],1,-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(inp.shape[0],1,-1,1)
                # sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
                # print(torch.sum(sub_sinogram - sinogram_eval))

            #lossTest
            # self.losses[i] = self.losses[i] + PSNR(sub_pet_eval.reshape(pet_gt1.shape),gt)
            self.losses[i] = self.losses[i] +mean_squared_error(sub_pet_eval.reshape(pet_gt1.shape),gt)
            if i == 0 or i == 10 or i == 20: #9 19 29
                x = sub_pet_eval.reshape(pet_gt1.shape)

                xo = x

                x = self.upnear(x)
                xinput = x
                ##Single

                pl = torch.Tensor([1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
                # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

                mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

                image_embeddings = self.model.image_encoder(x.repeat(1,3,1,1)*256)

                # Embed prompts
                multimask_output = False

                # last_x = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
                # last_y = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
                # xi = x[:,2-i:3-i,:,:]
                xi = x

                if self.use_point:
                    xstate2 = self.stateConv_p(self.statePool_p(xi)).reshape(xi.shape[0],256)
                    px = self.act1(xstate2)
                    py = self.act2(xstate2)
                    s_x = torch.argmax(px,dim=1)
                    s_y = torch.argmax(py,dim=1)
                    # last_x = s_x
                    # last_y = s_y
                    
                    point = torch.cat([s_x.unsqueeze(1).unsqueeze(1),s_y.unsqueeze(1).unsqueeze(1)],dim=2)

                    points_list.append(point)
                    points = (torch.cat(points_list,dim=1), pl[:,:ic+1])
                    ic = ic + 1
                else:
                    points = None

                if self.use_box:
                    #old
                    # xstate1 = self.stateConv(self.statePool(xi)).reshape(xi.shape[0],64)
                    # bx1,by1 = self.box1(xstate1)
                    # bx2,by2 = self.box2(xstate1)
                    # bx1 = torch.argmax(bx1,dim=1).unsqueeze(1)
                    # bx2 = torch.argmax(bx2,dim=1).unsqueeze(1)
                    # by1 = torch.argmax(by1,dim=1).unsqueeze(1)
                    # by2 = torch.argmax(by2,dim=1).unsqueeze(1)
                    #new
                    xstate1 = self.stateConv(self.statePool(xi)) # B,1,16,16
                    cx = torch.argmax(torch.sum(xstate1,dim=-1),dim=-1) * 10 /160*256 # -1 -2
                    cy = torch.argmax(torch.sum(xstate1,dim=-2),dim=-1) * 10 /160*256 # -2 -1
                    nozerox = torch.count_nonzero(xstate1>0.5,dim=-1)
                    nozeroy = torch.count_nonzero(xstate1>0.5,dim=-2)
                    squarex = nozerox.max(dim=-1)[0]*10# 面积等价 x y
                    squarey = nozeroy.max(dim=-1)[0]*10# 面积等价 y x
                    # print(squarex,squarey)
                    bx1 = cx + squarex *0.5
                    bx2 = cx - squarex *0.5
                    by1 = cy + squarey *0.5
                    by2 = cy - squarey *0.5
                    

                    pre_boxes = torch.cat([bx1,by1,bx2,by2],dim=1)

                else:
                    pre_boxes = None


                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=pre_boxes,
                    masks=mask_input,
                )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                masks = self.model.postprocess_masks(
                    low_res_masks,
                    input_size=x.shape[-2:],
                    original_size=256,
                )
                masks = torch.sigmoid(masks)
                # masks = (masks > 0.5).float()

                mask_list.append(masks)

                boxs_list.append(pre_boxes)

                sub_pet_eval = sub_pet_eval + self.sampledown(masks).reshape(inp.shape[0],1,-1,1)



        masks = torch.cat(mask_list, dim=1)


        y = masks
        y = self.downconv(self.sampledown(y))

        x = torch.cat([xo,y],dim=1)
        x = self.unet(x)

        if save_all:
            np.savez('./promptResPoint-EM-i1/batch-'+str(self.count)+'.npz',input=xinput.detach().cpu().numpy(),
                    sam_mask = masks.detach().cpu().numpy(),
                    point=torch.cat(points_list, dim=1).detach().cpu().numpy(),
                    box = torch.cat(boxs_list, dim=1).cpu().numpy(),
                    imask = mask_input.detach().cpu().numpy(),
                    sam_out = y.detach().cpu().numpy(),
                    output = x.detach().cpu().numpy(),
                    xo = xo.detach().cpu().numpy(),
                    gt = gt.detach().cpu().numpy())
            self.count = self.count+1
        return x,y
    




class SamAgentEmBr(nn.Module):
    def __init__(self,seve_all = False,use_box = True, use_point = True, unfreeze = False):
        super().__init__()
        args = Namespace()

        num_subsets = 1
        self.num_subsets = num_subsets
        self.total_iter = 30

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/mnt/c/sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        # self.unet = Unet(in_channels=self.total_iter//10+1, n_cls=1, n_filters=16)

        policy_hidden = 256
        self.act1 = PolicyNet(256, policy_hidden,256)
        self.act2 = PolicyNet(256,policy_hidden,256)
        self.box1 = PolicyBoxNet(64,policy_hidden,256)
        self.box2 = PolicyBoxNet(64,policy_hidden,256)

        # self.lin3 = nn.Linear(64,128)
        # self.boxhdim = nn.Linear(128,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.statePool = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.statePool_p = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv_p = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((128,128))

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.downconv = BasicConv2d(self.total_iter//10 , self.total_iter//10 , kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(1 , 1 , kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)
        
        self.count = 0



        det_count = 128
        all_angles = np.linspace(0,np.pi,128,False)
        self.subset_radons = []
        sub_len = int(len(all_angles) / num_subsets)
        self.sub_len = sub_len
        for i in range(num_subsets): 
            angles = all_angles[i*sub_len:(i+1)*sub_len]
            sub_radon = ParallelBeam(det_count,angles)
            self.subset_radons.append(sub_radon)


    def forward(self, inp,gt = None,save_all = False):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.unfreeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True

        sinogram = inp[:,0]
        mul_factor = inp[:,1]
        pet_noise = inp[:,2]

        num_subsets = self.num_subsets
        total_iter = self.total_iter
        pet_gt1 = torch.zeros((inp.shape[0],1,128,128)).cuda()
        sinogram = sinogram.reshape(pet_gt1.shape)
        mul_factor = mul_factor.reshape(pet_gt1.shape)
        pet_noise = pet_noise.reshape(pet_gt1.shape)

        subset_radons = self.subset_radons
        sub_len = self.sub_len

        # 迭代重建
        sub_pet_eval = torch.ones_like(pet_gt1).reshape(inp.shape[0],1,-1,1)


        mask_list = []
        points_list = []
        boxs_list = []

        ic = 0
        for i in range(total_iter):
            for subset in range(num_subsets):
                sub_sinogram = sinogram[:,:,subset*sub_len:(subset+1)*sub_len].reshape(inp.shape[0],1,-1,1)
                sub_mul_factor = mul_factor[:,:,subset*sub_len:(subset+1)*sub_len]
                sub_pet_noise = pet_noise[:,:,subset*sub_len:(subset+1)*sub_len]
                G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

                l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape))
                sinogram_eval = (sub_mul_factor.reshape(inp.shape[0],1,-1,1) * l.reshape(inp.shape[0],1,-1,1) + sub_pet_noise.reshape(inp.shape[0],1,-1,1))
                tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
                # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
                backproj = subset_radons[subset].backward((sub_mul_factor.reshape(inp.shape[0],1,-1,1)*tmp).reshape(sub_mul_factor.shape))
                sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(inp.shape[0],1,-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(inp.shape[0],1,-1,1)
                # sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
                # print(torch.sum(sub_sinogram - sinogram_eval))


            # if i == 0 or i == 10 or i == 20: #9 19 29 ;0 10 20 
            if i == 5:
                x = sub_pet_eval.reshape(pet_gt1.shape)

                xo = x

                x = self.upnear(x)
                xinput = x
                ##Single

                pl = torch.Tensor([1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
                # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

                mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

                image_embeddings = self.model.image_encoder(x.repeat(1,3,1,1)*256)

                # Embed prompts
                multimask_output = False

                # last_x = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
                # last_y = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
                # xi = x[:,2-i:3-i,:,:]
                xi = x

                if self.use_point:
                    xstate2 = self.stateConv_p(self.statePool_p(xi)).reshape(xi.shape[0],256)
                    px = self.act1(xstate2)
                    py = self.act2(xstate2)
                    s_x = torch.argmax(px,dim=1)
                    s_y = torch.argmax(py,dim=1)
                    # last_x = s_x
                    # last_y = s_y
                    
                    point = torch.cat([s_x.unsqueeze(1).unsqueeze(1),s_y.unsqueeze(1).unsqueeze(1)],dim=2)

                    points_list.append(point)
                    points = (torch.cat(points_list,dim=1), pl[:,:ic+1])
                    ic = ic + 1
                else:
                    points = None

                if self.use_box:
                    #old
                    # xstate1 = self.stateConv(self.statePool(xi)).reshape(xi.shape[0],64)
                    # bx1,by1 = self.box1(xstate1)
                    # bx2,by2 = self.box2(xstate1)
                    # bx1 = torch.argmax(bx1,dim=1).unsqueeze(1)
                    # bx2 = torch.argmax(bx2,dim=1).unsqueeze(1)
                    # by1 = torch.argmax(by1,dim=1).unsqueeze(1)
                    # by2 = torch.argmax(by2,dim=1).unsqueeze(1)
                    #new
                    xstate1 = self.stateConv(self.statePool(xi)) # B,1,16,16
                    cx = torch.argmax(torch.sum(xstate1,dim=-1),dim=-1) * 10 /160*256 # -1 -2
                    cy = torch.argmax(torch.sum(xstate1,dim=-2),dim=-1) * 10 /160*256 # -2 -1
                    nozerox = torch.count_nonzero(xstate1>0.5,dim=-1)
                    nozeroy = torch.count_nonzero(xstate1>0.5,dim=-2)
                    squarex = nozerox.max(dim=-1)[0]*10# 面积等价 x y
                    squarey = nozeroy.max(dim=-1)[0]*10# 面积等价 y x
                    # print(squarex,squarey)
                    bx1 = cx + squarex *0.5
                    bx2 = cx - squarex *0.5
                    by1 = cy + squarey *0.5
                    by2 = cy - squarey *0.5
                    

                    pre_boxes = torch.cat([bx1,by1,bx2,by2],dim=1)

                else:
                    pre_boxes = None


                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=pre_boxes,
                    masks=mask_input,
                )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                masks = self.model.postprocess_masks(
                    low_res_masks,
                    input_size=x.shape[-2:],
                    original_size=256,
                )
                masks = torch.sigmoid(masks)
                # masks = (masks > 0.5).float()

                mask_list.append(masks)

                boxs_list.append(pre_boxes)

                sub_pet_eval = sub_pet_eval + self.sampledown(masks).reshape(inp.shape[0],1,-1,1)

        masks = torch.cat(mask_list, dim=1)


        y = masks
        y = self.downconv(self.sampledown(y))

        x = torch.cat([xo,y],dim=1)
        # x = self.unet(x)
        x = self.conv1x1(x)
        x = torch.sigmoid(x)

        # if save_all:
        #     np.savez('./promptResPoint-EM-i1/batch-'+str(self.count)+'.npz',input=xinput.detach().cpu().numpy(),
        #             sam_mask = masks.detach().cpu().numpy(),
        #             point=torch.cat(points_list, dim=1).detach().cpu().numpy(),
        #             box = torch.cat(boxs_list, dim=1).cpu().numpy(),
        #             imask = mask_input.detach().cpu().numpy(),
        #             sam_out = y.detach().cpu().numpy(),
        #             output = x.detach().cpu().numpy(),
        #             xo = xo.detach().cpu().numpy(),
        #             gt = gt.detach().cpu().numpy())
        #     self.count = self.count+1
        return x,y
    




class SamAgentEmNew(nn.Module):
    def __init__(self,seve_all = False,use_box = True, use_point = True, unfreeze = True):
        super().__init__()
        args = Namespace()

        num_subsets = 1
        self.num_subsets = num_subsets
        self.total_iter = 30

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/mnt/c/sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        self.unet = Unet(in_channels=self.total_iter//10+1, n_cls=1, n_filters=16)

        self.act1 = PolicyNet(256,128,256)
        self.act2 = PolicyNet(256,128,256)
        self.box1 = PolicyBoxNet(64,128,256)
        self.box2 = PolicyBoxNet(64,128,256)

        self.lin3 = nn.Linear(64,128)
        self.boxhdim = nn.Linear(128,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.statePool = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.statePool_p = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv_p = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(self.total_iter//10 , self.total_iter//10 , kernel_size=3, stride=1, padding=1)
        
        self.count = 0



        det_count = 160
        all_angles = np.linspace(0,np.pi,160,False)
        self.subset_radons = []
        sub_len = int(len(all_angles) / num_subsets)
        self.sub_len = sub_len
        for i in range(num_subsets): 
            angles = all_angles[i*sub_len:(i+1)*sub_len]
            sub_radon = ParallelBeam(det_count,angles)
            self.subset_radons.append(sub_radon)


    def forward(self, inp,gt = None,save_all = False):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.unfreeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True

        sinogram = inp[:,0]
        mul_factor = inp[:,1]
        pet_noise = inp[:,2]

        num_subsets = self.num_subsets
        total_iter = self.total_iter
        pet_gt1 = torch.zeros((inp.shape[0],1,160,160)).cuda()
        sinogram = sinogram.reshape(pet_gt1.shape)
        mul_factor = mul_factor.reshape(pet_gt1.shape)
        pet_noise = pet_noise.reshape(pet_gt1.shape)

        subset_radons = self.subset_radons
        sub_len = self.sub_len

        # 迭代重建
        sub_pet_eval = torch.ones_like(pet_gt1).reshape(inp.shape[0],1,-1,1)


        mask_list = []
        points_list = []
        boxs_list = []

        ic = 0
        for i in range(total_iter):
            for subset in range(num_subsets):
                sub_sinogram = sinogram[:,:,subset*sub_len:(subset+1)*sub_len].reshape(inp.shape[0],1,-1,1)
                sub_mul_factor = mul_factor[:,:,subset*sub_len:(subset+1)*sub_len]
                sub_pet_noise = pet_noise[:,:,subset*sub_len:(subset+1)*sub_len]
                G_row_SumofCol = subset_radons[subset].backward(sub_mul_factor)

                l = subset_radons[subset].forward(sub_pet_eval.reshape(pet_gt1.shape))
                sinogram_eval = (sub_mul_factor.reshape(inp.shape[0],1,-1,1) * l.reshape(inp.shape[0],1,-1,1) + sub_pet_noise.reshape(inp.shape[0],1,-1,1))
                tmp = sub_sinogram / (sinogram_eval+torch.mean(sub_sinogram)*1e-9)
                # tmp[np.logical_and(sub_sinogram==0,sinogram_eval==0)]=1
                backproj = subset_radons[subset].backward((sub_mul_factor.reshape(inp.shape[0],1,-1,1)*tmp).reshape(sub_mul_factor.shape))
                sub_pet_eval = sub_pet_eval/(G_row_SumofCol.reshape(inp.shape[0],1,-1,1) +torch.mean(sub_pet_eval)*1e-9)*backproj.reshape(inp.shape[0],1,-1,1)
                # sub_pet_eval[G_row_SumofCol.reshape(-1,1)==0] = 0
                # print(torch.sum(sub_sinogram - sinogram_eval))


            if i == 5 or i == 15 or i == 25: #9 19 29
                x = sub_pet_eval.reshape(pet_gt1.shape)

                xo = x

                x = self.upnear(x)
                xinput = x
                ##Single

                pl = torch.Tensor([1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
                # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

                mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

                image_embeddings = self.model.image_encoder(x.repeat(1,3,1,1)*256)

                # Embed prompts
                multimask_output = False

                # last_x = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
                # last_y = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
                # xi = x[:,2-i:3-i,:,:]
                xi = x

                if self.use_point:
                    xstate2 = self.stateConv_p(self.statePool_p(xi)).reshape(xi.shape[0],256)
                    px = self.act1(xstate2)
                    py = self.act2(xstate2)
                    s_x = torch.argmax(px,dim=1)
                    s_y = torch.argmax(py,dim=1)
                    # last_x = s_x
                    # last_y = s_y
                    
                    point = torch.cat([s_x.unsqueeze(1).unsqueeze(1),s_y.unsqueeze(1).unsqueeze(1)],dim=2)

                    points_list.append(point)
                    points = (torch.cat(points_list[-1:],dim=1), pl[:,ic:ic+1])
                    ic = ic + 1
                else:
                    points = None

                if self.use_box:
                    #old
                    # xstate1 = self.stateConv(self.statePool(xi)).reshape(xi.shape[0],64)
                    # bx1,by1 = self.box1(xstate1)
                    # bx2,by2 = self.box2(xstate1)
                    # bx1 = torch.argmax(bx1,dim=1).unsqueeze(1)
                    # bx2 = torch.argmax(bx2,dim=1).unsqueeze(1)
                    # by1 = torch.argmax(by1,dim=1).unsqueeze(1)
                    # by2 = torch.argmax(by2,dim=1).unsqueeze(1)
                    #new
                    xstate1 = self.stateConv(self.statePool(xi)) # B,1,16,16
                    cx = torch.argmax(torch.sum(xstate1,dim=-1),dim=-1) * 10 /160*256 # -1 -2
                    cy = torch.argmax(torch.sum(xstate1,dim=-2),dim=-1) * 10 /160*256 # -2 -1
                    nozerox = torch.count_nonzero(xstate1>0.5,dim=-1)
                    nozeroy = torch.count_nonzero(xstate1>0.5,dim=-2)
                    squarex = nozerox.max(dim=-1)[0]*10# 面积等价 x y
                    squarey = nozeroy.max(dim=-1)[0]*10# 面积等价 y x
                    # print(squarex,squarey)
                    bx1 = cx + squarex *0.5
                    bx2 = cx - squarex *0.5
                    by1 = cy + squarey *0.5
                    by2 = cy - squarey *0.5
                    

                    pre_boxes = torch.cat([bx1,by1,bx2,by2],dim=1)

                else:
                    pre_boxes = None


                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=pre_boxes,
                    masks=mask_input,
                )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                masks = self.model.postprocess_masks(
                    low_res_masks,
                    input_size=x.shape[-2:],
                    original_size=256,
                )
                masks = torch.sigmoid(masks)
                # masks = (masks > 0.5).float()

                mask_list.append(masks)

                boxs_list.append(pre_boxes)

                sub_pet_eval = sub_pet_eval + self.sampledown(masks).reshape(inp.shape[0],1,-1,1)

        masks = torch.cat(mask_list, dim=1)

        res = sub_pet_eval.reshape(pet_gt1.shape)
        y = masks
        y = self.downconv(self.sampledown(y))

        x = self.sampledown(res)

        if save_all:
            np.savez('./promptResPoint-EMNew/batch-'+str(self.count)+'.npz',input=xinput.detach().cpu().numpy(),
                    sam_mask = masks.detach().cpu().numpy(),
                    point=torch.cat(points_list, dim=1).detach().cpu().numpy(),
                    box = torch.cat(boxs_list, dim=1).cpu().numpy(),
                    imask = mask_input.detach().cpu().numpy(),
                    sam_out = y.detach().cpu().numpy(),
                    output = x.detach().cpu().numpy(),
                    xo = xo.detach().cpu().numpy(),
                    gt = gt.detach().cpu().numpy())
            self.count = self.count+1
        return x,y


class SamAgentVL(nn.Module):
    def __init__(self,seve_all = False,use_box = True, use_point = True, unfreeze = True):
        super().__init__()
        args = Namespace()

        self.count = 0
        num_subsets = 1
        self.num_subsets = num_subsets
        self.total_iter = 30

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/mnt/c/sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        self.sampledown = nn.AdaptiveAvgPool2d((128,128))
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.downconv = BasicConv2d(self.total_iter//10 , self.total_iter//10 , kernel_size=3, stride=1, padding=1)
        self.unet = Unet(in_channels=4, n_cls=1, n_filters=16)




    def forward(self, inp,box,save_all=False,gt=None):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.unfreeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True

        x = inp

        x = self.upnear(x)
        ##Single

        # pl = torch.Tensor([1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
        # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

        mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

        image_embeddings = self.model.image_encoder(x.repeat(1,3,1,1)*256)

        # Embed prompts
        multimask_output = True

        points = None

            
        pre_boxes = box

        # pre_boxes = None


        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=pre_boxes,
            masks=mask_input,
        )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        masks = self.model.postprocess_masks(
            low_res_masks,
            input_size=x.shape[-2:],
            original_size=256,
        )
        masks = torch.sigmoid(masks)
        nx = self.sampledown(masks)
        # masks = (masks > 0.5).float()

        res = self.unet(torch.cat([nx,inp],dim=1))

        # x = self.sampledown(res)
        if save_all:
            np.savez('./promptResPoint-VL/batch-'+str(self.count)+'.npz',input=inp.detach().cpu().numpy(),
                    sam_mask = masks.detach().cpu().numpy(),
                    box = box.cpu().numpy(),
                    imask = nx.detach().cpu().numpy(),
                    output = res.detach().cpu().numpy(),
                    gt = gt.detach().cpu().numpy())
            self.count = self.count+1
        return res