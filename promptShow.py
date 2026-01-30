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
from MydataLoader import MyDatasetK,PETCTDataset,MyDatasetB
import argparse
from model import MLP,Unet
from EM import MLEMorKEM, normalize

import random
from torch_radon import ParallelBeam


import matplotlib.pyplot as plt

from multiprocessing import Pool

from samt import samt,samG,samb,samba,samf,samG3,samG3D



from metric import metric

# Training settings
parser = argparse.ArgumentParser(description='radio-VIT-implementation')
parser.add_argument('--batch_size', type=int, default=24, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=24, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=80, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=20, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

def process_EM(inputList):
    input,target,mul_factor,pet_noise = inputList
    # img = torch.zeros((3,128,128))
    pet_gt1 = normalize(target)*random.uniform(1,8)
    radon = ParallelBeam(128,np.linspace(0,180,128,False))
    kemresult = MLEMorKEM(input,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=15)
    # img[j][0] = torch.tensor(kemresult)
    # img[j][1] = torch.tensor(kemresult)
    # img[j][2] = torch.tensor(kemresult)
    return kemresult

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def run():
    models = ['EM-U','EM-SAM','EM-SAMF','EM-SAM-A','EM-U-SAM','EM-U-SAM-G','EM-U-P','EM-U-TP','EM-U-SAM-G3CP','EM-U-SAM-G3CPD-BNT2','EM-U-SAM-G3CPUF','EM-U-SAM-G3CP-Points','EM-U-SAM-G3CP-Boxs']
    model_name = models[9]
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
    lossL1 = nn.HuberLoss()

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

    if model_name == models[0]:
        net = Unet(in_channels=3, n_cls=1, n_filters=16).to(device)
    elif model_name == models[1]:
        net = samb().to(device)
    elif model_name == models[2]:
        net = samf().to(device)
    elif model_name == models[3]:
        net = samba().to(device)
    elif model_name == models[4]:
        net = samt().to(device)
    elif model_name == models[5]:
        net = samG().to(device)
    elif model_name == models[6]:
        net = Unet(in_channels=3, n_cls=1, n_filters=16).to(device)
    elif model_name == models[7]:
        net = Unet(in_channels=4, n_cls=1, n_filters=16).to(device)
    elif model_name == models[8]:
        net = samG3().to(device)
    elif model_name == models[10]:
        net = samG3(unfreeze=True,seve_all=True).to(device)
    elif model_name == models[9]:
        net = samG3D(seve_all=True).to(device)
    elif model_name == models[-2]:
        net = samG3(use_point=True,use_box=False,seve_all=True).to(device)
    elif model_name == models[-1]:
        net = samG3(use_point=False,use_box=True,seve_all=True).to(device)


    #继续训练
    # state_dict = torch.load('./RN50/VIT-ALL_ckt.pth', map_location="cpu")
    # net.load_state_dict(state_dict['network'])

    state_dict = torch.load('./RN50/'+model_name+'_ckt.pth')
    print(state_dict['network'])
    net.load_state_dict(state_dict['network'])
    # optimizer = optim.Adam(net.parameters(), lr=1e-3,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)


    # 加载数据集
    # train_dataset = MyDatasetK(is_train=True)
    # test_dataset = MyDatasetK(is_train=False)
    if model_name == models[7]:
        useCT = True
    else:
        useCT = False

    # test_dataset = PETCTDataset(is_train=False,use_ct=useCT)
    test_dataset = MyDatasetB(is_train=False)


    test_dataloader = DataLoader(test_dataset,batch_size=opt.test_batch_size,shuffle=False,num_workers=0,pin_memory=True)


    net.train()
    total_loss = 0


    ###################TEST#######################
    total_loss = 0
    total_psnr = 0
    qbar1 = trange(len(test_dataloader))
    net.eval()

    inputs = []
    preds = []
    targets = []

    for input,target in test_dataloader:

        with torch.no_grad():

            input = input.to(device)
            target = target.to(device)
            pred,_,_,_,_ = net(input,target)
            # pred = net(input,target)
            cur_loss = loss(pred,target)

            # if k == 3:


            
            psnr = metric(pred.detach().cpu().numpy(),target.detach().cpu().numpy())
        
            total_loss += cur_loss
            total_psnr  += psnr
            for i in range(len(input)):
                inputs.append(input[i].detach().cpu())
                preds.append(pred[i].detach().cpu())
                targets.append(target[i].detach().cpu())


        qbar1.set_postfix(epoch = 0,loss=cur_loss.item())  # 进度条右边显示信息
        # logger.info('[Test]epoch: {:.4f} Iteration: {:.4f} Loss:{:.4f}'.format(epoch,k, cur_loss.item()))
        qbar1.update(1)
    qbar1.close()
    np.savez('./result/'+model_name+'.npz',pred=preds,target=targets,input=inputs)
    vepoch_loss = total_loss / len(test_dataloader)  
    vepoch_psnr = total_psnr / len(test_dataloader) 
    print(vepoch_loss,vepoch_psnr)





if __name__ =='__main__':
    run()
