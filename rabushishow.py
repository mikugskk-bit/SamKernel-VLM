import numpy as np
import matplotlib.pyplot as plt



data = np.load('result\SAM-MED-EM-G.npz')

p = data['pred']
t = data['target']
print(data)


plt.subplot(3,2,1)
plt.imshow(p[116],cmap='gray')
plt.axis('off')
plt.subplot(3,2,2)
plt.imshow(t[116],cmap='gray')
plt.axis('off')
plt.subplot(3,2,3)
plt.imshow(p[1126],cmap='gray')
plt.axis('off')
plt.subplot(3,2,4)
plt.imshow(t[1216],cmap='gray')
plt.axis('off')
plt.subplot(3,2,5)
plt.imshow(p[16],cmap='gray')
plt.axis('off')
plt.subplot(3,2,6)
plt.imshow(t[16],cmap='gray')
plt.axis('off')
plt.show()