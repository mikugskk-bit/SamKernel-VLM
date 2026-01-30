import numpy as np
import matplotlib.pyplot as plt







data = np.load('./ctpetraw.npz')

ct = data['CT']
pet = data['PET']


for i in range(8000,len(ct),200):
    print(i)
    plt.subplot(1,2,1)
    plt.imshow(ct[i])
    plt.subplot(1,2,2)
    plt.imshow(pet[i])
    plt.show()
