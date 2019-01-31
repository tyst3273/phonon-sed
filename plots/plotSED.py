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

read = 'yes'
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
    
#    si = np.log(si)*np.log(si)*np.log(si)
#    ge = np.log(ge)*np.log(ge)*np.log(ge)
#    sl = np.log(sl)*np.log(sl)*np.log(sl)
#    
#    si = np.log(si)
#    ge = np.log(ge)
#    sl = np.log(sl)
    
#siN = si/si.max()*255
#geN = ge/ge.max()*255
#slN = sl/sl.max()*255

#bands = np.zeros((len(siN[:,0]),len(siN[0,:]),3),'uint8')
#bands[...,0] = siN
#bands[...,1] = geN
#bands[...,2] = slN
#
#img = Image.fromarray(bands,mode='RGB')
#img.save('bandsReg.png',format='png')
#img.show()


### PLOT ###
fig1, ax1 = plt.subplots()
ax1.imshow(np.log(si),interpolation='hamming',cmap='jet',aspect='equal')
#fig1.tight_layout()
#fig1.show()
fig1.savefig('si.png',dpi=2500,format='png',bbox_inches='tight',pad_inches=0.1)
             
fig2, ax2 = plt.subplots()
ax2.imshow(np.log(ge),interpolation='hamming',cmap='jet',aspect='equal')
#fig2.tight_layout()
#fig2.show()
fig2.savefig('ge.png',dpi=2500,format='png',bbox_inches='tight',pad_inches=0.1)

fig3, ax3 = plt.subplots()
ax3.imshow(np.log(sl),interpolation='hamming',cmap='jet',aspect='equal')
#fig3.tight_layout()
#fig3.show()
fig3.savefig('sl.png',dpi=2500,format='png',bbox_inches='tight',pad_inches=0.1)