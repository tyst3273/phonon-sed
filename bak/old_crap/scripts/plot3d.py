#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:57:07 2019

@author: ty
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import mod

#thz, kpoints, sed, dos = mod.readSED('si.cubic.2.dat')

maxf = 18
nf = len(sed[:,0])-np.argwhere(thz < maxf).max()
sed2 = mod.smoothSED(sed[nf:,:],0.0075,(thz[1,0]-thz[0,0])*1e12)

fig = plt.figure()
ax = fig.gca(projection='3d')

### Make data.
x = np.arange(0,len(kpoints[:,0]),1)
y = np.arange(0,len(sed2[:,0]),1)
x, y = np.meshgrid(x,y)

### Plot the surface.
surf = ax.plot_surface(x,y,sed2,cmap='jet',linewidth=0,antialiased=False)

### Customize the z axis.
ax.set_zlim(0, sed2.max())
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

### Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()