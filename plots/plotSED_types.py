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
    thz, kz, siz, dos = mod.readSED('SL.Si.inPlane.final.dat')
    thz, kx, six, dos = mod.readSED('SL.Si.crossPlane.final.dat')
    thz, kz, gez, dos = mod.readSED('SL.Ge.inPlane.final.dat')
    thz, kx, gex, dos = mod.readSED('SL.Ge.crossPlane.final.dat')
    thz, kz, slz, dos = mod.readSED('slInPlane.dat')
    thz, kx, slx, dos = mod.readSED('slCrossPlane.dat')
    
    kpoints = np.append(np.flipud(kx),kz,axis=0)
    del kx, kz
    
    ### REARRANGE ###
    six = np.fliplr(six[cutoff:-1,:])
    siz = np.real(siz[cutoff:-1,:])
    gex = np.fliplr(gex[cutoff:-1,:])
    gez = np.real(gez[cutoff:-1,:])    
    
    si = np.append(six,siz,axis=1)
    ge = np.append(gex,gez,axis=1)
    slT = np.real(np.append(np.fliplr(slx[cutoff:-1,:]),
                             slz[cutoff:-1,:],axis=1))
    
    sl = si[:,:]-ge[:,:]
#    sl = np.append(slT,sl,axis=1)
    
    ### SMOOTHE ###
    si = mod.smoothSED(si,win,(thz[1,0]-thz[0,0])*2e12*np.pi)
    ge = mod.smoothSED(ge,win,(thz[1,0]-thz[0,0])*2e12*np.pi)
#    sl = mod.smoothSED(sl,win,(thz[1,0]-thz[0,0])*2e12*np.pi)
    
    sl = sl*sl
    
    del win
    
#    si = np.log(si)*np.log(si)*np.log(si)
#    ge = np.log(ge)*np.log(ge)*np.log(ge)
#    sl = np.log(sl)*np.log(sl)*np.log(sl)
#    
#    six = np.log(six)
#    gex = np.log(gex)
#    siz = np.log(siz)
#    gez = np.log(gez)
#    
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
ax1.imshow(np.log(np.real(sl)),interpolation='hamming',cmap='jet',aspect='equal')
#fig1.tight_layout()
#fig1.show()
fig1.savefig('SL_both.png',dpi=2500,format='png',bbox_inches='tight',pad_inches=0.1)
             
#fig2, ax2 = plt.subplots()
#ax2.imshow(np.log(ge),interpolation='hamming',cmap='jet',aspect='equal')
##fig2.tight_layout()
##fig2.show()
#fig2.savefig('SL_ge.png',dpi=2500,format='png',bbox_inches='tight',pad_inches=0.1)

