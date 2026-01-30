import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft
import random

def _normalize_shape(x, d):
    old_shape = x.size()[:-d]
    x = x.view(-1, *(x.size()[-d:]))
    return x, old_shape
def _unnormalize_shape(y, old_shape):
    if isinstance(y, torch.Tensor):
        y = y.view(*old_shape, *(y.size()[1:]))
    elif isinstance(y, tuple):
        y = [yy.view(*old_shape, *(yy.size()[1:])) for yy in y]
    return y
def normalize_shape(d):
    def wrap(f):
        def wrapped(x, *args, **kwargs):
            x, old_shape = _normalize_shape(x, d)
            y = f(x, *args, **kwargs)
            return _unnormalize_shape(y, old_shape)
        return wrapped
    return wrap
import torch
import numpy as np
from torch.fft import rfft, irfft
class FourierFilters:
    def __init__(self):
        self.cache = dict()
    def get(self, size: int, filter_name: str, device):
        key = (size, filter_name)
        if key not in self.cache:
            ff = torch.FloatTensor(self.construct_fourier_filter(size, filter_name)).to(device)
            self.cache[key] = ff
        return self.cache[key].to(device)
    @staticmethod
    def construct_fourier_filter(size, filter_name):
        filter_name = filter_name.lower()

        n = torch.cat((torch.arange(1, size // 2 + 1, 2, dtype=torch.int),
                       torch.arange(size // 2 - 1, 0, -2, dtype=torch.int)))
        f = torch.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2

        fourier_filter = 2 * torch.real(rfft(f))
        if filter_name == "ramp" or filter_name == "ram-lak":
            pass
        elif filter_name == "shepp-logan":
            omega = np.pi * torch.fft.fftfreq(size)[1:]
            fourier_filter[1:] *= torch.sin(torch.Tensor(omega)) / torch.Tensor(omega)
        elif filter_name == "cosine":
            freq = torch.linspace(0, np.pi, size, endpoint=False)
            cosine_filter = torch.fft.fftshift(torch.sin(freq))
            fourier_filter *= cosine_filter
        elif filter_name == "hamming":
            fourier_filter *= torch.fft.fftshift(torch.from_numpy(np.hamming(size)))
        elif filter_name == "hann":
            fourier_filter *= torch.fft.fftshift(torch.from_numpy(np.hanning(size)))
        else:
            raise ValueError(f"Error, unknown filter type '{filter_name}', available filters are: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'")
        return fourier_filter[:size//2+1]
@normalize_shape(2)
def filter_sinogram(sinogram, fourier_filter=FourierFilters(),filter_name="ramp"):
    size = sinogram.size(2)
    n_angles = sinogram.size(1)

    # Pad sinogram to improve accuracy
    padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
    pad = padded_size - size
    padded_sinogram = torch.nn.functional.pad(sinogram.float(), (0, pad, 0, 0))

    sino_fft = rfft(padded_sinogram)

    # get filter and apply
    f = fourier_filter.get(padded_size, filter_name, sinogram.device)
    filtered_sino_fft = sino_fft * f.reshape(1, 1, -1)

    # Inverse fft
    filtered_sinogram = irfft(filtered_sino_fft)
    filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

    return filtered_sinogram.to(dtype=sinogram.dtype)


class Downsampler(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''
    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()
        
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1/2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1./np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'wrong name kernel'
            
            
        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        
        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch       

        self.downsampler_ = downsampler

        if preserve_size:

            if  self.kernel.shape[0] % 2 == 1: 
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)
                
            self.padding = nn.ReplicationPad2d(pad)
        
        self.preserve_size = preserve_size
        
    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x= input
        self.x = x
        return self.downsampler_(x)
        
def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']
    
    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box': 
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
    
        
    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1./(kernel_width * kernel_width)
        
    elif kernel_type == 'gauss': 
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'
        
        center = (kernel_width + 1.)/2.
        print(center, kernel_width)
        sigma_sq =  sigma * sigma
        
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center)/2.
                dj = (j - center)/2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos': 
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor  
                    dj = abs(j + 0.5 - center) / factor 
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor
                
                
                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)
                
                kernel[i - 1][j - 1] = val
            
        
    else:
        assert False, 'wrong method name'
    
    kernel /= kernel.sum()
    
    return kernel


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)
class UNet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True,ismaxval=True):
        super(UNet, self).__init__()
        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x
        self.ismaxval = ismaxval

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, norm_layer, need_bias, pad)

        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer, need_bias, pad)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer, need_bias, pad)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer, need_bias, pad)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer, need_bias, pad)

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels , norm_layer, need_bias, pad) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt =True) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)
        self.need_sigmoid = need_sigmoid
        if need_sigmoid: 
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, inputs,maxval=None):
        if maxval is None:
            maxval = torch.amax(inputs,dim=(1,2,3),keepdim=True)
            # if self.need_sigmoid:
            #     maxval = torch.amax(inputs,dim=(1,2,3),keepdim=True)
            # else:
            #     maxval = 1
        # inputs = inputs / maxval
        # Downsample
        downs = [inputs]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))

        in64 = self.start(inputs)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)

        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)

        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)

        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)

        down4 = self.down4(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                # print(prevs[-1].size())
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out,  downs[kk + 5]], 1)

                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_= l(up_, prevs[self.more - idx - 2])
        else:
            up_= down4

        up4= self.up4(up_, down3)
        up3= self.up3(up4, down2)
        up2= self.up2(up3, down1)
        up1= self.up1(up2, in64)
        if self.ismaxval:
            return self.final(up1)*maxval
        else:
            return self.final(up1)
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()
        if norm_layer is not None:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs= self.conv1(inputs)
        outputs= self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()
        self.conv= unetConv2(in_size, out_size, norm_layer, need_bias, pad)
        self.down= nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs= self.down(inputs)
        outputs= self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                   conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up= self.up(inputs1)
        
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2 
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2 
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output= self.conv(torch.cat([in1_up, inputs2_], 1))

        return output



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
    


class CNNBPNet(nn.Module):
    def __init__(self,radon,count=20e4) -> None:
        super().__init__()
        self.radon = radon
        self.count = count
        self.unet_recon = UNet(num_input_channels=3, num_output_channels=1,need_sigmoid=False)
        self.unet_prior = UNet(num_input_channels=2, num_output_channels=1,need_sigmoid=False)
    def forward(self, x,MR,mul_factor=None,noise=None,is_train=True):
        if is_train:
            return self.forward_train(x,MR)
        else:
            return self.forward_test(x,mul_factor,noise,MR)
    def forward_train(self,x,MR):
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
        x1 = self.unet_recon(torch.cat([x,mul_factor,noise],dim=1))
        x = filter_sinogram(x+x1)
        recon_alpha = self.radon.backward(x)
        prior = self.unet_prior(torch.cat([recon_alpha,MR],dim=1))
        img = (recon_alpha+prior)
        proj_out = self.radon.forward(img)
        sinogram_out = mul_factor*proj_out + noise
        return img,sinogram_out-sinogram
    def forward_test(self,sinogram,mul_factor,noise,mr):
        x = sinogram.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        mul_factor_t = mul_factor.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        noise_t = noise.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        mr_t = mr.reshape(np.prod(mr.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        x1 = self.unet_recon(torch.cat([x,mul_factor_t,noise_t],dim=1))
        x = filter_sinogram(x+x1)
        self.radon.forward(x)
        recon_alpha = self.radon.backward(x)
        prior = self.unet_prior(torch.cat([recon_alpha,mr_t],dim=1))
        img = recon_alpha+prior
        img = nn.ReLU()(img)
        return img
    


def _normalize_shape(x, d):
    old_shape = x.size()[:-d]
    x = x.view(-1, *(x.size()[-d:]))
    return x, old_shape
def _unnormalize_shape(y, old_shape):
    if isinstance(y, torch.Tensor):
        y = y.view(*old_shape, *(y.size()[1:]))
    elif isinstance(y, tuple):
        y = [yy.view(*old_shape, *(yy.size()[1:])) for yy in y]
    return y

class FBPNet_prior(nn.Module):
    def __init__(self,radon,count=20e4) -> None:
        super().__init__()
        self.radon = radon
        self.count=count
        self.unet = UNet(num_input_channels=2, num_output_channels=1,need_sigmoid=False)
        self.fourier_filters = nn.Parameter(FourierFilters().get(512,"ramp",torch.device("cuda")),requires_grad=True)
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
        filter_sino = self.filter_sinogram(nn.ReLU()((x-noise)/mul_factor),self.fourier_filters)
        recon_alpha = self.radon.backward(filter_sino)
        residual = self.unet(torch.concatenate([recon_alpha,MR],dim=1))
        img = recon_alpha + residual
        proj_out = self.radon.forward(img)
        sinogram_out = mul_factor*proj_out + noise
        return img,sinogram_out-sinogram
    def forward_test(self,sinogram,mul_factor,noise,mr=None):
        x = sinogram.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        mul_factor_t = mul_factor.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        noise_t = noise.reshape(np.prod(sinogram.shape)//(self.radon.volume.height*self.radon.volume.width),1,self.radon.volume.height,self.radon.volume.width)
        x = nn.ReLU()((x-noise_t)/mul_factor_t)
        x = self.filter_sinogram(nn.ReLU()((x-noise_t)/mul_factor_t),self.fourier_filters)
        self.radon.forward(x)#无效代码，仅仅是防止self.radon.backward(x)出错
        img = self.radon.backward(x)
        residual = self.unet(torch.concatenate([img,mr],dim=1))
        img = img + residual
        img = nn.ReLU()(img)
        return img
    def filter_sinogram(self,sinogram, fourier_filter=None):
        sinogram, old_shape = _normalize_shape(sinogram, 2)
        size = sinogram.size(2)
        n_angles = sinogram.size(1)

        # Pad sinogram to improve accuracy
        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
        pad = padded_size - size
        padded_sinogram = torch.nn.functional.pad(sinogram.float(), (0, pad, 0, 0))

        sino_fft = rfft(padded_sinogram)

        # get filter and apply
        f = fourier_filter
        filtered_sino_fft = sino_fft * f.reshape(1, 1, -1)

        # Inverse fft
        filtered_sinogram = irfft(filtered_sino_fft)
        filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))
        filtered_sinogram = filtered_sinogram.to(dtype=sinogram.dtype)
        filtered_sinogram =  _unnormalize_shape(filtered_sinogram, old_shape)
        return filtered_sinogram
    



def gaussian(M, std, sym=True):
    if M < 1:
        return torch.array([])
    if M == 1:
        return torch.ones(1)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w

def gaussiankern2D(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

def generate_random_image_torch(num_points=10):
    # Ensure the endpoints are included
    x = torch.tensor([0.0, 1.0])
    y = torch.tensor([0.0, 1.0])
    
    # Generate random x and y values
    x_random = torch.sort(torch.rand(num_points - 2))[0]
    y_random = torch.rand(num_points - 2)
    
    # Concatenate to include endpoints and sort
    x = torch.cat((x, x_random))
    y = torch.cat((y, y_random))
    sort_indices = torch.argsort(x)
    x = x[sort_indices]
    y = y[sort_indices]
    x = x.cuda()
    y = y.cuda()
    # Create a piecewise linear interpolant using PyTorch's interpolation
    def interp(x, xp, fp):
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)

        return m[indicies] * x + b[indicies]
    
    return lambda input_tensor:interp(input_tensor, x, y)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)
        return sasc_output

class MixedUnet(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='deconv', pad='zero', norm_layer=nn.InstanceNorm2d, need_sigmoid=False, need_bias=True):
        super(MixedUnet, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x


        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, norm_layer, need_bias, pad)
        self.start_prior = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, norm_layer, need_bias, pad)
        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer, need_bias, pad)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer, need_bias, pad)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer, need_bias, pad)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer, need_bias, pad)
        self.down1_prior = self.down1
        self.down2_prior = self.down2
        self.down3_prior = self.down3
        self.down4_prior = self.down4
        self.da1 = DANetHead(256,128,nn.InstanceNorm2d)
        self.da2 = DANetHead(128,64,nn.InstanceNorm2d)
        self.da3 = DANetHead(64,32,nn.InstanceNorm2d)
        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels , norm_layer, need_bias, pad) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt =True) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)

        if need_sigmoid: 
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, inputs,prior_img):
        meanval = torch.mean(inputs,dim=(-1,-2),keepdim=True)
        inputs = inputs/meanval
        downs = [inputs]
        downs_prior = [prior_img]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))
            downs_prior.append(down(downs_prior[-1]))

        in64 = self.start(inputs)
        in64_prior = self.start_prior(prior_img)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)
            in64_prior = torch.cat([in64_prior, downs_prior[0]], 1)
        down1 = self.down1(in64)
        down1_prior = self.down1_prior(in64_prior)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)
            down1_prior = torch.cat([down1_prior, downs_prior[1]], 1)
        down2 = self.down2(down1)
        down2_prior = self.down2_prior(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)
            down2_prior = torch.cat([down2_prior, downs_prior[2]], 1)
        down3 = self.down3(down2)
        down3_prior = self.down3_prior(down2_prior)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)
            down3_prior = torch.cat([down3_prior, downs_prior[3]], 1)
        down4 = self.down4(down3)
        down4_prior = self.down4_prior(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)
            down4_prior = torch.cat([down4_prior, downs_prior[4]], 1)
        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out,  downs[kk + 5]], 1)

                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_= l(up_, prevs[self.more - idx - 2])
        else:
            up_= down4

        up4= self.up4(up_, self.da1(torch.cat([down3,down3_prior],dim=1)))
        up3= self.up3(up4, self.da2(torch.cat([down2,down2_prior],dim=1)))
        up2= self.up2(up3, down1)
        up1= self.up1(up2, in64)

        return self.final(up1)*meanval

class IterativeNet(nn.Module):
    def __init__(self,radon,count=20e4) -> None:
        super().__init__()
        self.radon = radon
        self.count=count
        self.unet = MixedUnet(num_input_channels=1,num_output_channels=1)
        self.embedding = nn.Parameter(torch.randn(1,1,128,128))
        self.bias = nn.Parameter(torch.randn(1,1,128,128))
    def forward(self, x,MR=None,mul_factor=None,noise=None,is_train=True,is_random=False,count=5e4):
        if is_train:
            return self.forward_train(x,MR,is_random)
        else:
            return self.forward_test(x,MR)
    def one_iter(self,sinogram,mul_factor,pet_noise,G_row_SumofCol,pet_eval,radon):
        l = radon.forward(pet_eval)
        sinogram_eval = mul_factor*l + pet_noise
        tmp = sinogram/(sinogram_eval+torch.mean(sinogram)*1e-9)
        tmp[torch.logical_and(sinogram==0,sinogram_eval==0)]=1
        backproj = radon.backward(mul_factor*tmp)
        pet_eval = pet_eval/(G_row_SumofCol)*backproj
        pet_eval[G_row_SumofCol==0] = 0
        return pet_eval
    def forward_train(self,x,MR=None,is_random=False):
        proj = self.radon.forward(x)
        mul_factor = torch.ones_like(proj)
        mul_factor = mul_factor+(torch.rand_like(mul_factor)*0.2-0.1)
        noise = torch.ones_like(proj)*torch.mean(mul_factor*proj,dim=(-1,-2),keepdims=True) *0.2
        sinogram = mul_factor*proj + noise
        count = random.randint(self.count,self.count*10)
        cs = count/(1e-9+torch.sum(sinogram,dim=(-1,-2),keepdim=True))
        sinogram = sinogram*cs
        mul_factor = mul_factor*cs
        noise = noise*cs
        sinogram_noise = torch.poisson(sinogram)
        G_row_SumofCol = self.radon.backward(mul_factor)
        pet_eval_clean = torch.ones_like(x)
        pet_eval_noise = torch.ones_like(x)
        pet_gt = x
        num_points = random.randint(2,10)
        func = generate_random_image_torch(num_points)
        MR = func(MR.flatten()).reshape(MR.shape)
        loss = 0
        for i in range(30):
            pet_eval_clean = self.one_iter(sinogram,mul_factor,noise,G_row_SumofCol,pet_eval_clean,self.radon)
            pet_eval_noise = self.one_iter(sinogram_noise,mul_factor,noise,G_row_SumofCol,pet_eval_noise,self.radon)
            if is_random:
                if random.random()>0.4:
                    out = self.unet(pet_eval_noise,MR)+pet_eval_noise
                else:
                    out = self.unet(pet_eval_clean,MR)+pet_eval_clean
            else:
                out = self.unet(pet_eval_noise,MR)+pet_eval_noise
            loss += nn.HuberLoss()(out,pet_eval_clean)
        return pet_gt,loss
    def forward_test(self,noise_img,mr):
        out = self.unet(noise_img,mr)+noise_img
        return out
