import time
from EM import MLEMorKEM, normalize
from torch_radon import ParallelBeam
import random
import numpy as np
from torch.utils.data import DataLoader
from MydataLoader import MyDataset
import matplotlib.pyplot as plt
import torch

# def func2(args):  # multiple parameters (arguments)
#     # x, y = args
#     x = args[0]  # write in this way, easier to locate errors
#     y = args[1]  # write in this way, easier to locate errors

#     time.sleep(1)  # pretend it is a time-consuming operation
#     return x - y

def process_EM(args):
    input,target,mul_factor,pet_noise = args
    # img = torch.zeros((3,128,128))
    pet_gt1 = normalize(target)*random.uniform(1,8)
    radon = ParallelBeam(128,np.linspace(0,180,128,False))
    kemresult = MLEMorKEM(input,mul_factor,pet_noise,None,radon=radon,pet_gt=pet_gt1,method_index='KEM',prefix='KEM',total_iter=10)
    return kemresult


def run__pool():  # main process
    from multiprocessing import Pool

    cpu_worker_num = 16
    test_dataset = MyDataset(is_train=True)
    test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=0,pin_memory=True)
    for input,target,mul_factor,pet_noise in test_dataloader:
        process_args = []
        for b in range(64):
            iList = [input[b],target[b],mul_factor[b],pet_noise[b]]
            process_args.append(iList)
        with Pool(cpu_worker_num) as p:
            outputs = p.map(process_EM, process_args)

        outputs = torch.tensor(outputs)
        outputs = outputs.unsqueeze(1)
        outputs = outputs.repeat(1,3,1,1)

    return outputs
        



    '''Another way (I don't recommend)
    Using 'functions.partial'. See https://stackoverflow.com/a/25553970/9293137
    from functools import partial
    # from functools import partial
    # pool.map(partial(f, a, b), iterable)
    '''

if __name__ =='__main__':
    run__pool()