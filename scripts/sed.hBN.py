#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last Update: 05.06.2019

@author: Ty Sterling <ty.sterling@colorado.edu>
and
Riley Hadjis

"""

### GLOBAL VARIABLES ###
import numpy as np
import matplotlib.pyplot as plt
import mod

mod.tic()
mod.log('\n\thBN, 4 atom ortho cell\n\t05.06.2019\n'
        '\tnx, ny =  24x24, supercell = 1x1\n',new='yes')
#description of system configuration for record keeping

outfile = 'hBN.24x24'
velsfile = 'vels.dat'

n1, n2, n3 = [24,24,1] #size of simulation cell
u1, u2, u3 = [1,1,1] #size of supercell
ndk = 24 #k space mesh, number of points between speciak k points

#### This version only
natoms, ntypes, masses, pos = mod.readData('data.vels') 
pos = np.delete(pos,1,axis=1)
pos = np.delete(pos,2,axis=1)
####

split = 5 #times to split data for averaging
steps = 2**20 #total run time
dt = 1e-15 #lammps time step
dn = 2**3 #print frequency
prints = steps//dn #total times data is printed
tn = prints//split #times data is printed per chunk
thz = np.arange(0,tn)/(tn*dt*dn)*1e-12 #frequency in THz

#### GET POSITIONS AND UNITCELL INDICIES ###
num, pos2, masses2, uc, basis, ax, ay = mod.makehBN(n1,n2,u1,u2)
#get the positions from function

#### GET K POINTS ###
prim = np.array([[1,0,0],
                 [0,1,0],
                 [0,0,1]]).astype(float) #primitive lattice vectors

prim[:,0] = prim[:,0]*ax*u1 #a1
prim[:,1] = prim[:,1]*ay*u2 #a2
prim[:,2] = prim[:,2]*1 #a3

klabel = np.array(('G','X','Y','G'))
specialk = np.array([[0,0,0],   #G
                     [0.5,0,0], #X
                     [0,0.5,0], #Y
                     [0,0,0]])  #G

kpoints, kdist, nk = mod.makeKpoints(prim,specialk,ndk) #generate k points
mod.printParams(dt,dn,num,steps,split,nk,klabel,thz)
#print input data to screen and log file

#### GET VELOCITIES AND CALCULATE SED ###############
# You shouldn't have to change anything below here!!!
######################################################

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
        dos = mod.VDOS(dos,vels,tn,num,dt,dn,thz) 
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
                massarr = mod.getMasses(ids,pos,masses)
                
                vx = np.fft.fft(vels[:,ids,0].reshape(tn,nc)*exp
                                *massarr,axis=0)*dt*dn 
                vy = np.fft.fft(vels[:,ids,1].reshape(tn,nc)*exp
                                *massarr,axis=0)*dt*dn 
                vz = np.fft.fft(vels[:,ids,2].reshape(tn,nc)*exp
                                *massarr,axis=0)*dt*dn
                #scaling th FFT https://www.mathworks.com/matlabcentral/  ...
                #answers/15770-scaling-the-fft-and-the-ifft

                qdot[:,k] = (qdot[:,k]+(abs(vx.sum(axis=1))**2+
                    abs(vy.sum(axis=1))**2+abs(vz.sum(axis=1))**2)/nc)
                    
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


