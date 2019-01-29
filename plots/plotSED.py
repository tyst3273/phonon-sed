#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:58:29 2019

@author: ty
"""

import mod
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

read = 'no'
win = 0.1
cutoff = 2999

if read == 'yes':
    ### READ ###
    thz, kz, siz, dos = mod.readSED('siInPlane.dat')
    thz, kx, six, dos = mod.readSED('siCrossPlane.dat')
    thz, kz, gez, dos = mod.readSED('geInPlane.dat')
    thz, kx, gex, dos = mod.readSED('geCrossPlane.dat')
    thz, kz, slz, dos = mod.readSED('slInPlane.dat')
    thz, kx, slx, dos = mod.readSED('slCrossPlane.dat')
    
    kpoints = np.append(np.flipud(kx),kz,axis=0)
    del kx, kz
    
    ### REARRANGE ###
    si = np.real(np.append(np.fliplr(six[cutoff:-1,:]),siz[cutoff:-1,:],axis=1))
    ge = np.real(np.append(np.fliplr(gex[cutoff:-1,:]),gez[cutoff:-1,:],axis=1))
    sl = np.real(np.append(np.fliplr(slx[cutoff:-1,:]),
                             slz[cutoff:-1,:],axis=1))
    
    del six, siz, gex, gez, slx, slz, cutoff
    
    ### SMOOTHE ###
    si = mod.smoothSED(si,win,(thz[1,0]-thz[0,0])*2e12*np.pi)
    ge = mod.smoothSED(ge,win,(thz[1,0]-thz[0,0])*2e12*np.pi)
    sl = mod.smoothSED(sl,win,(thz[1,0]-thz[0,0])*2e12*np.pi)
    
    del win
    
siN = si/np.sum(si,axis=0).sum()
geN = ge/np.sum(ge,axis=0).sum()
slN = sl/np.sum(sl,axis=0).sum()

bands = np.zeros((len(siN[:,0]),len(siN[0,:]),3),'uint8')
bands[...,0] = siN*256
bands[...,1] = geN*256
bands[...,2] = slN*256

img = Image.fromarray(bands)
img.save('bands.png',format='png')
#img.show()


### PLOT ###
#plt.imshow(np.log(slN),interpolation='hamming',cmap='jet',aspect='auto')
#plt.tight_layout()
#plt.show()

