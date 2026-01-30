import os
from PIL import Image
import numpy as np

from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
# from vitM import ViT
from tqdm import *
import torchmetrics
from MydataLoader import MyDatasetK,PETCTDataset,MyDatasetRawB, MyDatasetRawH
import argparse
from model import MLP,Unet
from EM import MLEMorKEM, normalize

from cmodel import CNNBPNet, FBPNet_prior, ADMM_Net,IterativeNet

import random
from torch_radon import ParallelBeam


import matplotlib.pyplot as plt

from multiprocessing import Pool
from torch_radon import ParallelBeam
import matplotlib.pyplot as plt

import random
myradon = ParallelBeam(160,np.linspace(0, 180, 160, endpoint=False))



from metric import metric

# Training settings
parser = argparse.ArgumentParser(description='radio-VIT-implementation')
parser.add_argument('--batch_size', type=int, default=12, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=12, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=40, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=20, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()




def run():
    # models = ['CNNBPNet','FBPNet_prior','ADMMNet','IterativeNet']
    models = ['CNNBPNet-B','FBPNet_prior-B','ADMMNet-B','IterativeNet-B']

    model_name = models[2]
    model_path = "./RN50"

    logger.add(f"{model_path}/log"+model_name+".log")

    def get_scheduler(optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler


    # 创建损失函数
    loss = nn.L1Loss()
    loss_fn = nn.SmoothL1Loss()

    # 创建模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net =   VisionTransformer(
    #                 input_resolution=64,
    #                 patch_size=16,
    #                 width=768,
    #                 layers=12,
    #                 heads=12,
    #                 output_dim=11
    #             ).to(device)

    # net = model = ViT('B_16_imagenet1k', pretrained=True).to(device)
    # net = Unet(in_channels=3, n_cls=1, n_filters=16).to(device)
    # net = MLP().to(device)

    count = 200000
    if model_name == 'CNNBPNet' or model_name == 'CNNBPNet-B':
        net = CNNBPNet(myradon,count=count)
    elif model_name == 'FBPNet_prior' or model_name == 'FBPNet_prior-B':
        net = FBPNet_prior(myradon,count=count)
    elif model_name == 'ADMMNet' or model_name == 'ADMMNet-B':
        net = ADMM_Net(15,myradon,count=count)
    elif model_name == 'IterativeNet' or model_name == 'IterativeNet-B':
        net = IterativeNet(myradon,count=count)
    else:
        raise ValueError('Unknown model: ')
    
    net = net.to(device)

    model_name = model_name+str(count)+"HE"

    #继续训练
    # state_dict = torch.load('./RN50/VIT-ALL_ckt.pth', map_location="cpu")
    # net.load_state_dict(state_dict['network'])

    # state_dict = torch.load('J://B_16_imagenet1k.pth')
    # print(state_dict)
    # net.load_state_dict(state_dict)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
    scheduler = get_scheduler(optimizer, opt)

    # 加载数据集
    # train_dataset = MyDatasetK(is_train=True)
    # test_dataset = MyDatasetK(is_train=False)

    # train_dataset = MyDatasetRawB(is_train=True)
    # test_dataset = MyDatasetRawB(is_train=False)

    train_dataset = MyDatasetRawH(is_train=True)
    test_dataset = MyDatasetRawH(is_train=False)

    # train_dataset = HECKTORdataset(mode='train')
    # test_dataset = HECKTORdataset(mode='test')

    train_dataloader = DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=0,pin_memory=True)
    test_dataloader = DataLoader(test_dataset,batch_size=opt.test_batch_size,shuffle=False,num_workers=0,pin_memory=True)


    minpsnr = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        net.train()
        total_loss = 0

        qbar = trange(len(train_dataloader))
        for input,target,ct in train_dataloader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                input = input.to(device)
                target = target.to(device)
                ct = ct.to(device)
                pred,diff_sino = net(input,ct)

                # pred = pred.reshape(target.shape[0],target.shape[1],target.shape[2])
                cur_loss = loss_fn(pred,target)+loss_fn(diff_sino,torch.zeros_like(diff_sino))
                cur_loss.backward()
                # if k == 3:
                optimizer.step()


            
                total_loss += cur_loss

            qbar.set_postfix(epoch = epoch,loss=cur_loss.item())  # 进度条右边显示信息
            # logger.info('[Train]epoch: {:.4f} Iteration: {:.4f} Loss:{:.4f}'.format(epoch,k, cur_loss.item()))
            qbar.update(1)
        qbar.close()

        epoch_loss = total_loss / len(train_dataloader) 

        ###################TEST#######################
        total_loss = 0
        total_psnr = 0
        qbar1 = trange(len(test_dataloader))
        net.eval()

        inputs = []
        preds = []
        targets = []

        for input,target,ct in test_dataloader:
            optimizer.zero_grad()
            with torch.no_grad():

                input = input.to(device)
                target = target.to(device)
                ct = ct.to(device)
                pred,diff_sino = net(input,ct)

                # pred = pred.reshape(target.shape[0],target.shape[1],target.shape[2])
                cur_loss = loss_fn(pred,target)+loss_fn(diff_sino,torch.zeros_like(diff_sino))
                # if k == 3:
                optimizer.step()

                
                psnr = metric(pred,target)
            
                total_loss += cur_loss
                total_psnr  += psnr
                for i in range(len(input)):
                    inputs.append(input[i].detach().cpu())
                    preds.append(pred[i].detach().cpu())
                    targets.append(target[i].detach().cpu())


            qbar1.set_postfix(epoch = epoch,loss=cur_loss.item())  # 进度条右边显示信息
            # logger.info('[Test]epoch: {:.4f} Iteration: {:.4f} Loss:{:.4f}'.format(epoch,k, cur_loss.item()))
            qbar1.update(1)
        qbar1.close()
        vepoch_loss = total_loss / len(test_dataloader) 
        vepoch_psnr = total_psnr / len(test_dataloader) 
        


        if vepoch_psnr > minpsnr:
            minpsnr = vepoch_psnr
            checkpoint_path = f"{model_path}/{model_name}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"checkpoint_{epoch} saved")
            np.savez('./result/'+model_name+'.npz',pred=preds,target=targets,input=inputs)
        logger.info('Train Loss: {:.4f} Valid Loss: {:.4f} Valid PSNR:{:.4f}'.format(
            epoch_loss.item(), vepoch_loss.item(),vepoch_psnr.item()))
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)




if __name__ =='__main__':
    run()
