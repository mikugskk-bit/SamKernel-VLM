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

        # ds0 = self.block_1_2_left(self.block_1_1_left(x))
        # ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        # ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        # x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        # # x = self.block_5_2_left(self.block_5_1_left(self.pool_3(ds3)))

        # # x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(x), ds3], 1)))
        # x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        # x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        # x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)
        # return x
        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)

# class samG3D(nn.Module):
#     def __init__(self,seve_all = False):
#         super().__init__()
#         args = Namespace()

#         self.save_all = seve_all

#         args.image_size = 256
#         args.encoder_adapter = True
#         args.sam_checkpoint = r"F:\sam-med2d_b.pth"
#         self.model = sam_model_registry["vit_b"](args)
#         self.original_size = 256

#         self.unet = Unet(in_channels=9, n_cls=1, n_filters=16)

#         self.decoder = copy.deepcopy(self.model.mask_decoder)
#         self.lin11 = nn.Linear(256,2)
#         self.lin12 = nn.Linear(256,2)
#         self.lin13 = nn.Linear(256,2)

#         self.lin2 = nn.Linear(256,4)

#         self.lin31 = nn.Linear(256,4)
#         self.lin32 = nn.Linear(256,4)
#         self.lin33 = nn.Linear(256,4)

#         self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

#         self.sampleup = nn.AdaptiveMaxPool2d((256,256))
#         self.sampledown = nn.AdaptiveAvgPool2d((160,160))
#         # self.downconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
#         self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
#         self.maske = nn.AdaptiveMaxPool3d((1,64,64))
#         self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
#         self.downconv = BasicConv2d(6, 6, kernel_size=3, stride=1, padding=1)
        
#         self.count = 0


#     def forward(self, x,gt = None):
        
#         for param in self.model.parameters():
#             param.requires_grad = False
#         for param in self.decoder.parameters():
#             param.requires_grad = True
#         xo = x
#         # x1 = x[:,0:1,:,:].repeat(1,3,1,1)
#         # x2 = x[:,1:2,:,:].repeat(1,3,1,1)
#         # x3 = x[:,2:3,:,:].repeat(1,3,1,1)
#         # x = x.unsqueeze(1)
#         # x = x.repeat(1,3,1,1)
#         x = self.upconv(self.sampleup(x))

#         ##Single
#         pc1 = torch.sigmoid(self.lin11(torch.sum(x[:,0,:,:],dim=1))) * 256
#         pc2 = torch.sigmoid(self.lin12(torch.sum(x[:,1,:,:],dim=1))) * 256
#         pc3 = torch.sigmoid(self.lin13(torch.sum(x[:,2,:,:],dim=1))) * 256
 
#         box1 = torch.sigmoid(self.lin31(torch.sum(x[:,0,:,:],dim=1)))  * 256
#         box2 = torch.sigmoid(self.lin32(torch.sum(x[:,1,:,:],dim=1)))  * 256
#         box3 = torch.sigmoid(self.lin33(torch.sum(x[:,2,:,:],dim=1)))  * 256

#         # print(pc1,pc2,pc3)

#         pc = torch.cat([pc1.unsqueeze(1),pc2.unsqueeze(1),pc3.unsqueeze(1)],dim=1)
#         pl = torch.Tensor([0,1,2]).unsqueeze(0).repeat(x.shape[0],1)
#         box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

#         mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

#         image_embeddings = self.model.image_encoder(x)


#         mask_list = []
#         # Embed prompts
#         multimask_output = False
#         for i in range(3):
#             points = (pc, pl)
#             pre_boxes = box[:,i:i+1]



#             sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#                 points=points,
#                 boxes=pre_boxes,
#                 masks=mask_input,
#             )  # sparse_embeddings:[2, 3, 256],  dense_embeddings:[2, 256, 64, 64]

#             low_res_masks, iou_predictions = self.model.mask_decoder(
#                 image_embeddings=image_embeddings,
#                 image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=multimask_output,
#             )

#             low_res_masks2, iou_predictions2 = self.decoder(
#                 image_embeddings=image_embeddings,
#                 image_pe=self.model.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=multimask_output,
#             )

#             masks = self.model.postprocess_masks(
#                 low_res_masks,
#                 input_size=x.shape[-2:],
#                 original_size=256,
#             )

#             masks2 = self.model.postprocess_masks(
#                 low_res_masks2,
#                 input_size=x.shape[-2:],
#                 original_size=256,
#             )

#             mask_list.append(masks)
#             mask_list.append(masks2)

#         masks = torch.cat(mask_list, dim=1)

#         # print(masks.shape)

#         # inputs1 = {'image':x,'original_size':256,'point_coords':pc,'point_labels':pl,"boxes":box,"mask_inputs":mask}
#         # outputs1 = self.model(inputs1,True)
#         # inputs2 = {'image':x[:,1:2,:,:],'original_size':256,'point_coords':pc[:,1:2],'point_labels':pl[:,1:2],"boxes":box2,"mask_inputs":mask}
#         # outputs2 = self.model(inputs2,False)
#         # inputs3 = {'image':x[:,2:3,:,:],'original_size':256,'point_coords':pc[:,2:3],'point_labels':pl[:,2:3],"boxes":box3,"mask_inputs":mask}
#         # outputs3 = self.model(inputs3,False)
#         # y = torch.cat([outputs1["masks"],outputs2["masks"],outputs3["masks"]],dim=1)
#         # sam_mask = y
#         y = masks
#         y = self.downconv(self.sampledown(y))

#         x = torch.cat([xo,y],dim=1)
#         x = self.unet(x)

#         if self.save_all:
#             np.savez('./promptRes/batch-'+str(self.count)+'.npz',input=xo.detach().cpu().numpy(),
#                     sam_mask = masks.detach().cpu().numpy(),
#                     point=pc.detach().cpu().numpy(),
#                     box = box.detach().cpu().numpy(),
#                     imask = mask_input.detach().cpu().numpy(),
#                     sam_out = y.detach().cpu().numpy(),
#                     output = x.detach().cpu().numpy(),
#                     xo = xo.detach().cpu().numpy(),
#                     gt = gt.detach().cpu().numpy())
#             self.count = self.count+1
#         return x,y


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

class samG3D(nn.Module):
    def __init__(self,seve_all = False,use_box = True, use_point = True, unfreeze = False):
        super().__init__()
        args = Namespace()

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = r"F:\sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        self.unet = Unet(in_channels=6, n_cls=1, n_filters=16)

        self.act1 = PolicyNet(256,128,256)
        self.act2 = PolicyNet(256,128,256)
        self.box1 = PolicyBoxNet(64,128,256)
        self.box2 = PolicyBoxNet(64,128,256)

        self.lin3 = nn.Linear(64,128)
        self.boxhdim = nn.Linear(128,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.statePool = nn.AdaptiveMaxPool3d((1,8,8))
        self.stateConv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.statePool_p = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv_p = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        isize = 128
        self.sampledown = nn.AdaptiveAvgPool2d((isize,isize))
        self.sampledown2 = nn.AdaptiveMaxPool2d((isize,isize))

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        self.count = 0

        self.decoder = copy.deepcopy(self.model.mask_decoder)


    def forward(self, x,gt = None):
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = True

        xo = x

        # x = self.upconv(self.sampleup(x))
        x = self.upnear(x)
        xinput = x
        ##Single

        pl = torch.Tensor([1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
        # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

        mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

        image_embeddings = self.model.image_encoder(x*256)


        mask_list = []
        mask_list2 = []
        points_list = []
        boxs_list = []

        # lp= torch.Tensor([128,128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
        # points_list.append(lp.unsqueeze(1))
        # Embed prompts
        multimask_output = False
        for i in range(3):
            # last_x = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
            # last_y = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
            xi = x[:,2-i:3-i,:,:]

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
                xstate1 = self.stateConv(self.statePool(xi)).reshape(xi.shape[0],64)
                bx1,by1 = self.box1(xstate1)
                bx2,by2 = self.box2(xstate1)
                bx1 = torch.argmax(bx1,dim=1).unsqueeze(1)
                bx2 = torch.argmax(bx2,dim=1).unsqueeze(1)
                by1 = torch.argmax(by1,dim=1).unsqueeze(1)
                by2 = torch.argmax(by2,dim=1).unsqueeze(1)
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

            low_res_masks2, iou_predictions2 = self.decoder(
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

            masks2 = self.model.postprocess_masks(
                low_res_masks2,
                input_size=x.shape[-2:],
                original_size=256,
            )
            masks = torch.sigmoid(masks)
            masks2 = torch.sigmoid(masks2)
            masks = (masks > 0.5).float()

            mask_list.append(masks)
            mask_list2.append(masks2)

            boxs_list.append(pre_boxes)

        masks = torch.cat(mask_list, dim=1)
        masks2 = torch.cat(mask_list2, dim=1)

        z = masks
        y = masks2
        y = self.downconv(self.sampledown(y))  # unfreeze mask
        z = self.sampledown2(z)  # freeze mask

        x = torch.cat([xo,y],dim=1)
        x = self.unet(x)

        if self.save_all:
            np.savez('./promptRes/batch-'+str(self.count)+'.npz',input=xinput.detach().cpu().numpy(),
                    sam_mask = masks.detach().cpu().numpy(),
                    sam_mask2 = masks2.detach().cpu().numpy(),
                    point=torch.cat(points_list, dim=1).detach().cpu().numpy(),
                    box = torch.cat(boxs_list, dim=1).cpu().numpy(),
                    imask = mask_input.detach().cpu().numpy(),
                    sam_out = y.detach().cpu().numpy(),
                    output = x.detach().cpu().numpy(),
                    xo = xo.detach().cpu().numpy(),
                    gt = gt.detach().cpu().numpy())
            self.count = self.count+1
        return x,y,z,points,torch.cat(boxs_list, dim=1).cpu().numpy()



class samG3(nn.Module):
    def __init__(self,seve_all = True,use_box = True, use_point = True, unfreeze = False):
        super().__init__()
        args = Namespace()

        self.use_box = use_box
        self.use_point = use_point
        self.save_all = seve_all
        self.unfreeze = unfreeze

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = r"F:\sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.original_size = 256

        self.unet = Unet(in_channels=6, n_cls=1, n_filters=16)

        self.act1 = PolicyNet(256,128,256)
        self.act2 = PolicyNet(256,128,256)
        self.box1 = PolicyBoxNet(64,128,256)
        self.box2 = PolicyBoxNet(64,128,256)

        self.lin3 = nn.Linear(64,128)
        self.boxhdim = nn.Linear(128,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.statePool = nn.AdaptiveMaxPool3d((1,8,8))
        self.stateConv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.statePool_p = nn.AdaptiveMaxPool3d((1,16,16))
        self.stateConv_p = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))

        self.upnear = nn.UpsamplingNearest2d(size=(256,256))
        # self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        self.count = 0


    def forward(self, x,gt = None):
        
        for param in self.model.parameters():
            param.requires_grad = False
        if self.unfreeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True
        xo = x

        # x = self.upconv(self.sampleup(x))
        x = self.upnear(x)
        xinput = x
        ##Single

        pl = torch.Tensor([1,1,1]).unsqueeze(0).repeat(x.shape[0],1)
        # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

        mask_input = torch.sigmoid(self.maskconv(self.maske(x)))

        image_embeddings = self.model.image_encoder(x*256)


        mask_list = []
        points_list = []
        boxs_list = []

        # lp= torch.Tensor([128,128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
        # points_list.append(lp.unsqueeze(1))
        # Embed prompts
        multimask_output = False
        for i in range(3):
            # last_x = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
            # last_y = torch.Tensor([128]).unsqueeze(0).repeat(x.shape[0],1).cuda()
            xi = x[:,2-i:3-i,:,:]

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
                xstate1 = self.stateConv(self.statePool(xi)).reshape(xi.shape[0],64)
                bx1,by1 = self.box1(xstate1)
                bx2,by2 = self.box2(xstate1)
                bx1 = torch.argmax(bx1,dim=1).unsqueeze(1)
                bx2 = torch.argmax(bx2,dim=1).unsqueeze(1)
                by1 = torch.argmax(by1,dim=1).unsqueeze(1)
                by2 = torch.argmax(by2,dim=1).unsqueeze(1)
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

        masks = torch.cat(mask_list, dim=1)


        y = masks
        y = self.downconv(self.sampledown(y))

        x = torch.cat([xo,y],dim=1)
        x = self.unet(x)

        if self.save_all:
            np.savez('./promptResBAP/batch-'+str(self.count)+'.npz',input=xinput.detach().cpu().numpy(),
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





class samG(nn.Module):
    def __init__(self):
        super().__init__()
        args = Namespace()

        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = r"F:\sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.conv = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1)

        self.unet = Unet(in_channels=6, n_cls=1, n_filters=16)


        self.lin1 = nn.Linear(256,2)

        self.lin2 = nn.Linear(256,4)

        self.lin3 = nn.Linear(256,4)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))
        # self.downconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.downconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        self.count = 0



    def forward(self, x,gt):
        
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        xo = x
        # x1 = x[:,0:1,:,:].repeat(1,3,1,1)
        # x2 = x[:,1:2,:,:].repeat(1,3,1,1)
        # x3 = x[:,2:3,:,:].repeat(1,3,1,1)
        # x = x.unsqueeze(1)
        # x = x.repeat(1,3,1,1)
        x = self.upconv(self.sampleup(x))

        ##Single
        pc1 = torch.sigmoid(self.lin1(torch.sum(x[:,0,:,:],dim=1))) * 256
        pc2 = torch.sigmoid(self.lin1(torch.sum(x[:,1,:,:],dim=1))) * 256
        pc3 = torch.sigmoid(self.lin1(torch.sum(x[:,2,:,:],dim=1))) * 256
 
        box = torch.sigmoid(self.lin3(torch.sum(x[:,1,:,:],dim=1)))  * 256

        pc = torch.cat([pc1.unsqueeze(1),pc2.unsqueeze(1),pc3.unsqueeze(1)],dim=1)
        pl = torch.Tensor([0,1,2]).unsqueeze(0).repeat(x.shape[0],1)
        # box = torch.cat([box1.unsqueeze(1),box2.unsqueeze(1),box3.unsqueeze(1)],dim=1)

        mask = self.maskconv(self.maske(x))

        x_original = x
        inputs = {'image':x,'original_size':256,'point_coords':pc,'point_labels':pl,"boxes":box,"mask_inputs":mask}
        outputs = self.model(inputs,True)
        y = outputs["masks"]
        sam_mask = y

        y = self.downconv(self.sampledown(y))

        x = torch.cat([xo,y],dim=1)
        x = self.unet(x)


        np.savez('./promptRes/batch-'+str(self.count)+'.npz',input=x_original.detach().cpu().numpy(),
                sam_mask = sam_mask.detach().cpu().numpy(),
                point=pc.detach().cpu().numpy(),
                box = box.detach().cpu().numpy(),
                imask = mask.detach().cpu().numpy(),
                sam_out = y.detach().cpu().numpy(),
                output = x.detach().cpu().numpy(),
                xo = xo.detach().cpu().numpy(),
                gt = gt.detach().cpu().numpy())
        self.count = self.count+1
        return x,y
    


class samt(nn.Module):
    def __init__(self):
        super().__init__()
        args = Namespace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = r"F:\sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)
        self.conv = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1)

        self.unet = Unet(in_channels=6, n_cls=1, n_filters=16)

        self.sampleup = nn.AdaptiveMaxPool2d((256,256))

        self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.sampledown = nn.AdaptiveAvgPool2d((160,160))




    def forward(self, x):
        for param in self.model.parameters():
            param.requires_grad = False
        xo = x
        # x1 = x[:,0:1,:,:].repeat(1,3,1,1)
        # x2 = x[:,1:2,:,:].repeat(1,3,1,1)
        # x3 = x[:,2:3,:,:].repeat(1,3,1,1)
        # x = x.unsqueeze(1)
        # x = x.repeat(1,3,1,1)
        x = self.upconv(self.sampleup(x))

        inputs = {'image':x,'original_size':256}
        outputs = self.model(inputs,True)
        y = outputs["masks"]

        y = self.sampledown(y)


        x = torch.cat([xo,y],dim=1)
        x = self.unet(x)


        return x
    
class samb(nn.Module):
    def __init__(self):
        super().__init__()
        args = Namespace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = r"F:\sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))
        self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)




    def forward(self, x):
        # for param in self.model.parameters():
        #     param.requires_grad = False
        xo = x

        x = self.upconv(self.sampleup(x))

        inputs = {'image':x,'original_size':256}
        outputs = self.model(inputs,True)
        y = outputs["masks"]

        y = self.sampledown(y)

        x = torch.cat([xo,y],dim=1)

        y = self.conv1(y)
        y = torch.sigmoid(y)

        return y
    

class samf(nn.Module):
    def __init__(self):
        super().__init__()
        args = Namespace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = r"F:\sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))
        self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)




    def forward(self, x):
        for param in self.model.parameters():
            param.requires_grad = False
        xo = x

        x = self.upconv(self.sampleup(x))

        inputs = {'image':x,'original_size':256}
        outputs = self.model(inputs,True)
        y = outputs["masks"]

        y = self.sampledown(y)

        x = torch.cat([xo,y],dim=1)

        y = self.conv1(y)
        y = torch.sigmoid(y)

        return y
    
class samba(nn.Module):
    def __init__(self):
        super().__init__()
        args = Namespace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = r"F:\sam-med2d_b.pth"
        self.model = sam_model_registry["vit_b"](args)

        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

        self.sampleup = nn.AdaptiveMaxPool2d((256,256))
        self.sampledown = nn.AdaptiveAvgPool2d((160,160))
        self.upconv = BasicConv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.lin1 = nn.Linear(256,2)

        self.lin2 = nn.Linear(256,4)

        self.lin3 = nn.Linear(256,4)

        self.maske = nn.AdaptiveMaxPool3d((1,64,64))
        self.maskconv = BasicConv2d(1, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        for param in self.model.parameters():
            param.requires_grad = False
        xo = x

        x = self.upconv(self.sampleup(x))
        label = [ord('P'),ord('C'),ord('M'),ord('b')]

        ##Single
        pc = torch.sigmoid(self.lin1(torch.sum(torch.sum(x,dim=1),dim=1))) * 256
        index = torch.argmax(torch.softmax(self.lin2(torch.sum(torch.sum(x,dim=1),dim=1)),dim=1),dim=1)
        pl = torch.Tensor([label[i] for i in index])
        box = torch.softmax(self.lin3(torch.sum(torch.sum(x,dim=1),dim=1)),dim=1)
        mask = self.maskconv(self.maske(x))
        pc = pc.unsqueeze(1)
        pl = pl.unsqueeze(1)
        box = box.unsqueeze(1)
        inputs = {'image':x,'original_size':256,'point_coords':pc,'point_labels':pl,"boxes":box,"mask_inputs":mask}
        outputs = self.model(inputs,True)
        y = outputs["masks"]

        y = self.sampledown(y)

        x = torch.cat([xo,y],dim=1)

        y = self.conv1(y)
        y = torch.sigmoid(y)

        return y
