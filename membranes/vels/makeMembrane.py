#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:36:03 2019

@author: ty
"""

def makeMem(nx,lammps='no'):
    """
    Same as makeSL except return a & c, lattice constants for x=y and 
    z respectively
    """
    import numpy as np
    import copy as cp
    
    with open('memUnit.xyz','r') as fid:
        num = int(fid.readline())
        fid.readline()
        pos = np.zeros((num,3))
        for i in range(num):
            tmp = fid.readline().strip().split()
            pos[i,0] = tmp[1]
            pos[i,1] = tmp[2]
            pos[i,2] = tmp[3]
        pos = pos[np.argsort(pos[:,0]),:]
            
    basis = cp.deepcopy(pos)
    basis = basis[np.argsort(basis[:,0]),:]
    del pos
    
    masses = np.array([28.0855]) 
    a = 31.23+1.36 #lattice constant
    
    uc = np.zeros(num)
    ids = np.arange(0,num)
    
    pos = cp.deepcopy(basis)
    tmp = cp.deepcopy(basis)
    tuc = cp.deepcopy(uc)
    tmpids = cp.deepcopy(ids)
    for i in range(nx-1):
        tmp[:,0] = tmp[:,0]+a
        pos = np.append(pos,tmp,axis=0)
        tuc = tuc+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tmpids,axis=0)
    
    num = len(pos)
    pos = np.append(np.ones((num,1)),pos,axis=1)
    pos = np.append(np.arange(1,num+1).reshape(num,1),pos,axis=1)
    
    if lammps != 'no':    
        with open(lammps,'w') as fid:
            xbuff = 1.36/2.0
            ybuff = 1.36/2.0
            zbuff = 20
            fid.write('LAMMPS DATA FILE FOR SED\n')
            fid.write('\n'+str(num)+' atoms\n')
            fid.write('\n'+str(len(masses))+' atom types\n')
            fid.write('\n'+str(0-xbuff)+' '+
                      str(((31.23+1.36)*nx-1.36)+xbuff)+' xlo xhi\n')
            fid.write(str(0-ybuff)+' '+str(31.23+ybuff)+
                      ' ylo yhi\n')
            fid.write(str(pos[:,4].min()-zbuff)+' '+str(pos[:,4].max()+zbuff)+
                      ' zlo zhi\n')
            fid.write('\nMasses\n\n')
            for i in range(len(masses)):
                fid.write(str(i+1)+' '+str(masses[i])+'\n')
            fid.write('\nAtoms\n\n')
            for i in range(num-1):
                fid.write(str(int(pos[i,0]))+' '+str(int(pos[i,1]))+' '
                          +str(pos[i,2])+' '+str(pos[i,3])+' '+
                          str(pos[i,4])+'\n')
            fid.write(str(int(pos[-1,0]))+' '+str(int(pos[-1,1]))+' '
                      +str(pos[-1,2])+' '+str(pos[-1,3])+' '+str(pos[-1,4]))

    return [num, pos, masses, uc, ids, a]    