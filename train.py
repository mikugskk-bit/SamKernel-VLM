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
from MydataLoader import MyDataset
import argparse
from model import MLP,Unet
from EM import MLEMorKEM, normalize

import random
from torch_radon import ParallelBeam


import matplotlib.pyplot as plt

from multiprocessing import Pool

from samt import samt,samG

import cv2

# Training settings
parser = argparse.ArgumentParser(description='radio-VIT-implementation')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=80, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate for adam')
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
    model_name = "SAM-MED-EM-G3P"
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

    net = samG().to(device)


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
    train_dataset = MyDataset(is_train=True)
    test_dataset = MyDataset(is_train=False)

    train_dataloader = DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=0,pin_memory=True)
    test_dataloader = DataLoader(test_dataset,batch_size=opt.test_batch_size,shuffle=False,num_workers=0,pin_memory=True)




    cpu_worker_num = 16
    minloss = 100
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        net.train()
        total_loss = 0
        
        #CPU - EM - train
        #init temp
        temp = []
        for input,target,mul_factor,pet_noise in train_dataloader:
            temp.append(input)

        for k in range(1):
            #CPU - EM - train
            process_args = []
            for input,train in zip(temp,train_dataloader):
                for b in range(len(input)):
                    iList = [input[b],train[1][b],train[2][b],train[3][b]]
                    process_args.append(iList)
            with Pool(cpu_worker_num) as p:
                outputs = p.map(process_EM, process_args)

            batch_outputs = split_batch(outputs,opt.batch_size)


            qbar = trange(len(train_dataloader))
            temp = []
            for output,train in zip(batch_outputs,train_dataloader):
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):

                    image = np.array(output)
                    new_image = np.zeros([len(image),256,256])
                    for i in range(len(image)):
                        new_image[i]  = cv2.resize(image[i], (256, 256), interpolation=cv2.INTER_CUBIC)
                    output = torch.tensor(new_image).float()

                    # output = output.unsqueeze(1)
                    # output = output.repeat(1,3,1,1)
                    imgt = output.to(device)
                    pred = net(imgt)
                    target = train[1]
                    input = train[0]

                    temp.append(pred.reshape(input.shape[0],input.shape[1],input.shape[2]).detach().cpu())

                    pred = pred.reshape(target.shape[0],target.shape[1],target.shape[2])
                    target = target.to(device)
                    cur_loss = loss(pred,target)
                    cur_loss.backward()
                    # if k == 3:
                    optimizer.step()
                
                    total_loss += cur_loss

                qbar.set_postfix(epoch = epoch,loss=cur_loss.item(),iter = k)  # 进度条右边显示信息
                # logger.info('[Train]epoch: {:.4f} Iteration: {:.4f} Loss:{:.4f}'.format(epoch,k, cur_loss.item()))
                qbar.update(1)
            qbar.close()

        epoch_loss = total_loss / len(train_dataloader) 


        ###################TEST#######################
        qbar1 = trange(len(test_dataloader))
        net.eval()

        #CPU - EM - test
        temp = []
        for input,target,mul_factor,pet_noise in test_dataloader:
            temp.append(input)

        for k in range(1):
            #CPU - EM - test
            process_args = []
            for input,train in zip(temp,test_dataloader):
                for b in range(len(input)):
                    iList = [input[b],train[1][b],train[2][b],train[3][b]]
                    process_args.append(iList)
            with Pool(cpu_worker_num) as p:
                outputs = p.map(process_EM, process_args)

            batch_outputs = split_batch(outputs,opt.batch_size)

            with torch.no_grad():
                total_loss = 0
                qbar1 = trange(len(test_dataloader))
                temp = []
                for output,train in zip(batch_outputs,test_dataloader):

                    image = np.array(output)
                    new_image = np.zeros([len(image),256,256])
                    for i in range(len(image)):
                        new_image[i]  = cv2.resize(image[i], (256, 256), interpolation=cv2.INTER_CUBIC)

                    output = torch.tensor(new_image).float()
                    # output = output.unsqueeze(1)
                    # output = output.repeat(1,3,1,1)
                    imgt = output.to(device)
                    pred = net(imgt)
                    input = train[0]

                    temp.append(pred.reshape(input.shape[0],input.shape[1],input.shape[2]).detach().cpu())

                    target = train[1]            
                    pred = pred.reshape(target.shape[0],target.shape[1],target.shape[2]).detach().cpu()
                    cur_loss = loss(pred,target)
                    total_loss += cur_loss

                    qbar1.set_postfix(epoch = epoch,loss=cur_loss.item())  # 进度条右边显示信息
                    # logger.info('[Test]epoch: {:.4f} Iteration: {:.4f} Loss:{:.4f}'.format(epoch,k, cur_loss.item()))
                    qbar1.update(1)
                qbar1.close()
            vepoch_loss = total_loss / len(test_dataloader) 
        


        if vepoch_loss < minloss:
            minloss = vepoch_loss
            checkpoint_path = f"{model_path}/{model_name}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"checkpoint_{epoch} saved")
        logger.info('Train Loss: {:.4f} Valid Loss: {:.4f}'.format(
            epoch_loss.item(), vepoch_loss.item()))
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        np.savez('./result/'+model_name+'.npz',pred=pred,target=target,input=output)



if __name__ =='__main__':
    run()
