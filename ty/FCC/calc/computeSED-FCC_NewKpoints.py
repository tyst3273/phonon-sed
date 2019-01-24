#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Date Stamp: 01.17.2019

@author: Ty Sterling <ty.sterling@colorado.edu>
and
Riley Hadjis

This was written in python 2.7 installed with conda.
"""

### GLOBAL VARIABLES ###
import numpy as np #if it doesnt work, lemme know and ill tell you the versions
import matplotlib.pyplot as plt
import mod

mod.tic()

outfile = 'sed.ty.Si'
velsfile = 'vels.dat'

n1, n2, n3 = [24,24,24] #size of simulation cell
dk = 30 #k space mesh, number of points between speciak k points

steps = 50000 #run time
dt = 0.5e-15 #lammps time step
dn = 20 #print frequency
prints = steps/dn #times data is printed
split = 1 #times to split data for averaging
tn = prints/split #timesteps per chunk
win = 0.75 #gaussian smoothing window
pi = np.pi #tired of forgetting the 'np' part...

#om = np.arange(0,tn)*2*np.pi/(tn*dt*dn) #angular frequency
thz = np.arange(0,tn)/(tn*dt*dn)*1e-12 #frequency in THz

#### GET POSITIONS AND UNITCELL INDICIES ###
num, pos, masses, uc, a = mod.makeTriclinic(n1,n2,n3,element='si') 
#get the positions from function
types = pos[:,1]
###

#### GET K POINTS ###
prim = np.array([[2,0,0],
                 [1,3,0],
                 [1,1,1]]).astype(float) #primitive lattice vectors

prim[:,0] = prim[:,0]*a*np.sqrt(2.0)/4.0 #a1
prim[:,1] = prim[:,1]*a/np.sqrt(24.0) #a2
prim[:,2] = prim[:,2]*a/np.sqrt(3.0) #a3

#specialk = np.array([[0,0,0], #gamma
#                     [0.5,0,0.5], #X
#                     [0.75,0.375,0.375], #K
#                     [0,0,0], #gamma
#                     [0.5,0.5,0.5]]) #L 
#                     #special reciprocal lattice points
#klabel = np.array(('G','X','K','G','L')) 

specialk = np.array([[0,0,0], #gamma
                     [0.5,0.5,0], #X
                     [0.75,0.5,0.25], #W
                     [0.75,0.375,0.375], #K
                     [0,0,0], #G
                     [0.5,0.5,0.5], #L
                     [0.625,0.625,0.25], #U
                     [0.75,0.5,0.25], #W
                     [0.5,0.5,0.5], #L
                     [0.75,0.375,0.375]]) #K
                     #special reciprocal lattice points
klabel = np.array(('G','X','W','K','G','L','U','W','L','K')) 

#same k-path as 'Materials Project'
#<https://materialsproject.org/materials/mp-149/>

kpoints, kdist = mod.makeKpoints(prim,specialk,dk) #get the input k space array
#from a funtion

### GET VELOCITIES AND CALCULATE SED ###
mod.log('\n\tCALCULATING PHONON SPECTRAL ENERGY DENSITY!',new='yes')

with open(velsfile, 'r') as fid: 
    nk = len(kpoints) #number of k points
    ids = np.zeros((num))
    nc = max(uc)+1
    nb = len(masses)
    ids = np.argwhere(types == 1) #atoms in this basis pos
    cellvec = pos[ids[:,0],:] #coords of fcc basis atom
    cellvec = cellvec[np.argsort(uc[ids][:,0]),:] #sort by unit cell
    
    sed = np.zeros((tn,nk)) #spectral energy density
    dos = np.zeros((tn,1))
    
    #the data is read in in chunks and the chunks are mathed upon until
    #its a smaller data structure. then the chunks are all block averaged
    #together. Saves RAM space and it also 'ensemble averages' to 
    #produce better data      
          
    for i in range(split): #loop over chunks to block average
        mod.log('\n\tNow on chunk: '+str(i+1)+
              ' out of '+str(split)+'\n')
        vels = np.zeros((tn,num,3))
        qdot = np.zeros((tn,nk))
        
        #read in vels for this block
        for j in range(tn): #loop over timesteps in block
            tmpVels = np.zeros((num,3))
            for k in range(9): #skip comments
                fid.readline()
            for k in range(num): #get atoms
                tmp = fid.readline().strip().split()
                vels[j,k,0] = float(tmp[2]) #vx
                vels[j,k,1] = float(tmp[3]) #vy
                vels[j,k,2] = float(tmp[4]) #vz
         
        #calculate vibrational density of states
#        vdos = mod.vdos(vels,tn,num,dt,dn,win,thz) #wasn't working
#        dos[:,0] = dos[:,0]+vdos[0][:]
                
        #compute SED for this block        
        for j in range(nk): #loop over all k points
            kvec = kpoints[j,:] #k point vector
            if j%10 == 0:
                mod.log('\t\tNow on k-point: '+str(j)+' out of '+str(nk)+'\n')
            tmp = np.zeros((tn,3)).astype(complex) #sed for this k point
            for k in range(nc-1): #loop over unit cells
                rvec = cellvec[k,2:5] #position of unit cell
                ids = np.argwhere(uc==k) 
                for l in range(nb): #loop over basis atoms
                    mass = masses[l]
                    vx = vels[:,ids[l,0],0] #time series for particular atom
                    vy = vels[:,ids[l,0],1] # '' ''
                    vz = vels[:,ids[l,0],2] # '' ''
                    sfft = np.exp(1j*np.dot(kvec,rvec))
                    
                    tmp[:,0] = tmp[:,0]+vx*sfft
                    tmp[:,1] = tmp[:,1]+vy*sfft
                    tmp[:,2] = tmp[:,2]+vz*sfft
                    #space fft of velocity data
                    tmpfft = np.fft.fft(tmp,axis=0)
                    qdot[:,j] = np.multiply(abs(tmpfft),
                        abs(tmpfft)).sum(axis=1)*mass 
                    #KE of normal coordinate (square of time-FFT)
                    
        sed = sed+np.real(qdot)/(4*np.pi*steps/split*dt*nc) #a buncha constants
        mod.writeSED(outfile+'.'+str(i)+'.dat',thz,kpoints,sed/(i+1),dos/(i+1))
        mod.toc() #execution time
        
sed = sed/split #average across splits
dos = dos/split

del cellvec, i, j, k, ids, l, mass, n1, n2, n3, nb, nc, pi, prints, qdot
del split, steps, tmp, tmpVels, types, uc, vels, velsfile, vx, vy, vz
#clean up variables

### WRITE TO A FILE ###
mod.writeSED(outfile+'.final.dat',thz,kpoints,sed,dos)

sedg = mod.smoothSED(sed,win,(thz[1]-thz[0])*2*np.pi*1e12)
#gaussian smooth SED along freq axis for better looking results
mod.writeSED(outfile+'.smooth.dat',thz,kpoints,sedg,dos)

mod.log('\n\tAll done!')

### PLOT THE DISPERSION CURVE ###
plt.imshow(np.log(sedg),interpolation='hamming',cmap='jet',aspect='auto')
plt.tight_layout()
plt.show()
###


