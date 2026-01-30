import torch
from skimage.data import shepp_logan_phantom
import numpy as np
from torch_radon import ParallelBeam
import matplotlib.pyplot as plt

all_angles = np.linspace(0,np.pi,400,False)
angles1 = all_angles[0:200]
angles2 = all_angles[200:-1]
pet_gt1 = shepp_logan_phantom()
# pet_gt1 = pad_and_center_image(pet_gt1,(image_size,image_size))
# 投影矩阵
radon1 = ParallelBeam(400,angles1)
radon2 = ParallelBeam(400,angles2)
radon = ParallelBeam(400,all_angles)

sino = radon.forward(torch.from_numpy(pet_gt1).float().cuda())
x_gt = radon.backward(sino)

# sino1 = radon1.forward(torch.from_numpy(pet_gt1).float().cuda())
sino1 = sino[:200]
x1 = radon1.backward(sino1)
# sino2 = radon2.forward(torch.from_numpy(pet_gt1).float().cuda())
sino2 = sino[200:]
x2 = radon2.backward(sino2)

plt.subplot(1,4,1)
plt.imshow(sino1.detach().cpu().numpy())
plt.subplot(1,4,2)
plt.imshow(x1.detach().cpu().numpy())
plt.subplot(1,4,3)
plt.imshow(sino2.detach().cpu().numpy())
plt.subplot(1,4,4)
plt.imshow(x2.detach().cpu().numpy())
plt.show()

plt.subplot(1,2,1)
plt.imshow((x1+x2).detach().cpu().numpy())
plt.subplot(1,2,2)
plt.imshow((x_gt).detach().cpu().numpy())
plt.show()