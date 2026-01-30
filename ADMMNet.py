import torch
import torch.nn as nn
import torch.nn.functional as F
class XUpdateBlock(nn.Module):
    def __init__(self,radon):
        super(XUpdateBlock,self).__init__()
        self.rho = torch.nn.Parameter(torch.tensor(0.003),requires_grad=True)
        self.t = torch.nn.Parameter(torch.tensor(0.0001),requires_grad=True)
        self.radon = radon
    def forward(self,ins):
        #
        x = ins[0]
        y = ins[1] #torch.unsqueeze(ins[1],1)
        z = ins[2]
        b = ins[3]
        #-----
        Ax = self.radon.forward(x) - y
        Ayh =self.radon.backward(Ax) * self.t
        #-----
        x1 = (1-self.t*self.rho)*x + self.t*self.rho*(z-b)-self.t*Ayh
        #
        return (x1,y,z,b)

class ZUpdateBlock(nn.Module):
    def __init__(self,channels=16):
        super(ZUpdateBlock,self).__init__()
        self.rho = torch.nn.Parameter(torch.tensor(0.003),requires_grad=True)
        self.k = torch.nn.Parameter(torch.tensor(0.0001),requires_grad=True)

        self.conv_i1o32 = nn.Conv2d(1,channels,3,padding=1)
        self.conv_i32o32 = nn.Conv2d(channels,channels,3,padding=1)
        self.conv_i32o1 = nn.Conv2d(channels,1,3,padding=1)
    def forward(self,ins):
        #
        x = ins[0]
        y = ins[1] #torch.unsqueeze(ins[1],1)
        z = ins[2]
        b = ins[3]
        #-----
        R = self.conv_i1o32(z)
        R = F.leaky_relu(self.conv_i32o32(R))
        R = F.leaky_relu(self.conv_i32o1(R))
        #-----
        z1 = (1-self.k*self.rho)*z + self.k*self.rho*(x+b)-self.k*R
        #
        return (x,y,z1,b)

class InterBlock(nn.Module):
    def __init__(self,radon):
        super(InterBlock,self).__init__()
        self.eta = torch.nn.Parameter(torch.tensor(0.0001),requires_grad=True)
        
        self.layers_up_x = []
        for i in range(15):
            self.layers_up_x.append(XUpdateBlock(radon))
        self.net_x =  nn.Sequential(*self.layers_up_x)

        self.layers_up_z = []
        for i in range(7):
            self.layers_up_z.append(ZUpdateBlock())
        self.net_z =  nn.Sequential(*self.layers_up_z)
        
    def forward(self,ins):
        #
        x = ins[0]
        y = ins[1] #torch.unsqueeze(ins[1],1)
        z = ins[2]
        b = ins[3]
        #-----x
        [x1,y,z,b] = self.net_x([x,y,z,b])
        #------z
        [x1,y,z1,b] = self.net_z([x1,y,z,b])
        #-------b
        b1 = b + self.eta * (x1 - z1)
        #       
        return (x1,y,z1,b1)
        
import numpy as np

class ADMM_Net(nn.Module):
    def __init__(self,n_iter,radon,count=20e4):
        super(ADMM_Net, self).__init__()
        self.radon = radon
        self.n_iter = n_iter
        self.layers = []
        self.count=  count
        for i in range(n_iter):
            self.layers.append(InterBlock(radon))
        self.net =  nn.Sequential(*self.layers)
        # self.b0 = torch.nn.Parameter(torch.tensor(0.003),requires_grad=True).cuda()
        # self.b0 = torch.tensor(0.0003,requires_grad=False).cuda()
    def forward(self, x,MR=None,mul_factor=None,noise=None,is_train=True):
        if is_train:
            return self.forward_train(x,MR)
        else:
            return self.forward_test(x,mul_factor,noise,MR)
    def forward_train(self,x,MR=None):
        proj = self.radon.forward(x)
        mul_factor = torch.ones_like(proj)
        mul_factor = mul_factor+(torch.rand_like(mul_factor)*0.2-0.1)
        noise = torch.ones_like(proj)*torch.mean(mul_factor*proj,dim=(-1,-2),keepdims=True) *0.2
        sinogram = mul_factor*proj + noise
        cs = self.count/(1e-9+torch.sum(sinogram,dim=(-1,-2),keepdim=True))
        sinogram = sinogram*cs
        mul_factor = mul_factor*cs
        noise = noise*cs
        x = torch.poisson(sinogram)
        x = nn.ReLU()((x-noise)/mul_factor)
        [img,yst,zst,bst] = self.net((torch.zeros_like(x),x,torch.zeros_like(x),torch.zeros_like(x)))
        proj_out = self.radon.forward(img)
        sinogram_out = mul_factor*proj_out + noise
        return img,sinogram_out-sinogram
    def forward_test(self,sinogram,mul_factor,noise,mr=None):
        x = sinogram.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        mul_factor_t = mul_factor.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        noise_t = noise.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        x = nn.ReLU()((x-noise_t)/mul_factor_t)
        [img,yst,zst,bst] = self.net((torch.zeros_like(x),x,torch.zeros_like(x),torch.zeros_like(x)))
        return img