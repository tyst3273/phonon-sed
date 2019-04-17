#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Stamp: 04.15.2019

@author: Ty Sterling <ty.sterling@colorado.edu>
and
Riley Hadjis

Significantly reworked. Namely, there is no longer a loop over
unit cells. Instead, all unit cells are treated at once for each
basis atom.

Previously, I was erroneously summing over unit cells and
adding the x, y, and z velocities BEFORE taking the
absolute-value-square of the velocity FFT's... which is wrong.

This version seems to match lattice dynamics results quite well!
"""

### GLOBAL VARIABLES ###
import numpy as np #if it doesnt work, lemme know and ill tell you the versions
import matplotlib.pyplot as plt
import mod

mod.tic()
mod.log('\n\tBulk silicon, 8 atom cubic cell\n\t04.16.2019,2**24 steps\n',new='yes')
#description of system configuration for record keeping

outfile = 'si.cubic'
velsfile = 'vels.dat'

n1, n2, n3 = [50,4,4] #size of simulation cell
nk = 50 #k space mesh, number of points between speciak k points

split = 4 #times to split data for averaging
steps = 2**24 #total run time
dt = 1e-15 #lammps time step
dn = 2**4 #print frequency
prints = steps//dn #total times data is printed
tn = prints//split #times data is printed per chunk
thz = np.arange(0,tn)/(tn*dt*dn)*1e-12 #frequency in THz

#### GET POSITIONS AND UNITCELL INDICIES ###
num, pos, mass, uc, basis, a = mod.makeSi(n1,n2,n3)
#get the positions from function

#### GET K POINTS ###
prim = np.array([[1,0,0],
                 [0,1,0],
                 [0,0,1]]).astype(float) #primitive lattice vectors

prim[:,0] = prim[:,0]*a #a1
prim[:,1] = prim[:,1]*a #a2
prim[:,2] = prim[:,2]*a #a3

klabel = np.array(('G','X'))
specialk = np.array([[0,0,0], #G
                     [0.5,0,0]]) #X 

kpoints, kdist = mod.makeKpoints(prim,specialk,nk) #generate k points
mod.printParams(dt,dn,num,steps,split,nk,klabel,thz)
#print input data to screen and log file

#### GET VELOCITIES AND CALCULATE SED 
sed = np.zeros((tn,nk)) #spectral energy density
dos = np.zeros((tn,4)) #total, x, y, z
with open(velsfile, 'r') as fid: 
    ids = np.zeros((num))
    nc = max(uc).astype(int)+1 #number of unit cells
    nb = len(np.unique(basis)) #number of basis atoms
    ids = np.argwhere(basis == 0) #reference atom for each unit cell
    cellvec = pos[ids[:,0],:] #coords of unit cells
    
    for i in range(split): #loop over chunks to block average
        mod.log('\n\tNow on chunk: '+str(i+1)+
              ' out of '+str(split)+'\n')
        vels = np.zeros((tn,num,3))
        qdot = np.zeros((tn,nk))

        mod.log('\t\tNow reading velocities...\n')
        for j in range(tn): #loop over timesteps in block
            if j!= 0 and j%(tn//10) == 0:
                mod.log('\t\t'+str(j/(tn//10)*10)+'% done reading '
                        'velocites')
            tmpVels = np.zeros((num,3))
            for k in range(9): #skip comments
                fid.readline()
            for k in range(num): #get atoms
                tmp = fid.readline().strip().split()
                vels[j,k,0] = float(tmp[2]) #vx
                vels[j,k,1] = float(tmp[3]) #vy
                vels[j,k,2] = float(tmp[4]) #vz

        ##calculate vibrational density of states
        mod.log('\n\t\tNow computing vibrational density of states...\n')
#        dos = mod.VDOS(dos,vels,tn,num,dt,dn,thz) 
        mod.toc()
                
        ##compute SED for each block
        mod.log('\n\t\tNow computing spectral energy density...\n')        
        for k in range(nk): #loop over all k points
            kvec = kpoints[k,:] #k point vector
            if k%10 == 0:
                mod.log('\t\tNow on k-point: '+str(k)+' out of '+str(nk))
            exp = mod.getExp(nc,kvec,cellvec)
            
            for b in range(nb): #loop over basis atoms
                ids = np.argwhere(basis == b)
                vx = np.fft.fft(vels[:,ids,0].reshape(tn,nc)*exp,axis=0)*dt*dn 
                vy = np.fft.fft(vels[:,ids,1].reshape(tn,nc)*exp,axis=0)*dt*dn 
                vz = np.fft.fft(vels[:,ids,2].reshape(tn,nc)*exp,axis=0)*dt*dn
                #scaling th FFT https://www.mathworks.com/matlabcentral/  ...
                #answers/15770-scaling-the-fft-and-the-ifft

                qdot[:,k] = (qdot[:,k]+mass/nc*(abs(vx.sum(axis=1))**2+
                    abs(vy.sum(axis=1))**2+abs(vz.sum(axis=1))**2))
                    
        sed = sed+qdot/(4*np.pi*dt*tn*dn) #a buncha constants
        mod.writeSED(outfile+'.'+str(i)+'.dat',thz,kpoints,sed/(i+1),dos/(i+1))
        mod.toc()
        
sed = sed/split #average across splits
dos = dos/split

#### WRITE TO A FILE ###
mod.writeSED(outfile+'.final.dat',thz,kpoints,sed,dos)
mod.log('\n\tAll done!')
mod.toc()

#### PLOT THE DISPERSION CURVE ###
plt.imshow(np.log(sed),interpolation='hamming',cmap='jet',aspect='auto')
plt.tight_layout()
plt.show()
####


