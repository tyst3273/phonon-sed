#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is Ty's custom module to for the SED code: keep it in the same
directory

DATE STAMP: 02.20.2019 
DATE STAMP: 04.15.2019
DATE STAMP: 05.01.2019

Updated to initialize hBN monolayers with supercell basis.
"""
import numpy as np
import sys
import copy as cp
############################################################
def gsmooth(Raw, win, dom):
    """
    This function takes an numpy array defined over some domain and returns
    a gaussian smoothened version. "Raw" is the input numpy array. "win" is 
    the width of the gaussian smoothing window used to initialize the 
    smoothing window; e.g. if you have a signal from 0 to 20 THz and you want
    to smooth with a window of 1/3 THz, set "win = 1/3.0". dom is the constant
    spacing between values in the domain; e.g. if you have a numpy array,
    "freq", ranging from 0 to 20 THz and with length 10000, "dom = freq[1] - 
    freq[0]" = 0.002 THz. If none of this makes sense, do like I did and
    figure it our yourself ;)
    """
    gwin = round(win*1e12*2*np.pi/dom) #number of array elements in window
    if gwin % 2 == 0: #make sure its odd sized array
        gwin = gwin+1
    if gwin == 1:
        gauss = np.array([1]) #if array is size 1, convolve with self
    else:
        n = 2*np.arange(0,gwin)/(gwin-1)-(1) #centered at 0, sigma = 1
        n = (3*n) #set width of profile to 6*sigma i.e. end values ~= 0
        gauss = np.exp(-np.multiply(n,n)/2.0) #gaussian fx to convolve with
        gauss = gauss/np.sum(gauss) #normalized gaussian

    smooth = np.convolve(Raw,gauss,mode='same')
    return smooth

####################################################################
def smoothSED(Raw, win, dom):
    """
    Same as gsmooth but smooths 2d array along axis=0; designed for smoothing
    SED.
    This function takes an numpy array defined over some domain and returns
    a gaussian smoothened version. "Raw" is the input numpy array. "win" is 
    the width of the gaussian smoothing window used to initialize the 
    smoothing window; e.g. if you have a signal from 0 to 20 THz and you want
    to smooth with a window of 1/3 THz, set "win = 1/3.0". dom is the constant
    spacing between values in the domain; e.g. if you have a numpy array,
    "freq", ranging from 0 to 20 THz and with length 10000, "dom = freq[1] - 
    freq[0]" = 0.002 THz. If none of this makes sense, do like I did and
    figure it our yourself ;)
    """
    gwin = round(win*1e12*2*np.pi/dom) #number of array elements in window
    if gwin % 2 == 0: #make sure its odd sized array
        gwin = gwin+1
    if gwin == 1:
        gauss = np.array([1]) #if array is size 1, convolve with self
    else:
        n = 2*np.arange(0,gwin)/(gwin-1)-(1) #centered at 0, sigma = 1
        n = (3*n) #set width of profile to 6*sigma i.e. end values ~= 0
        gauss = np.exp(-np.multiply(n,n)/2.0) #gaussian fx to convolve with
        gauss = gauss/np.sum(gauss) #normalized gaussian
    
    smooth = np.zeros((len(Raw[:,0]),len(Raw[0,:])))
    for i in range(len(Raw[0,:])):
        smooth[:,i] = np.convolve(Raw[:,i],gauss,mode='same')
    return smooth

###############################################################
def readData(filename):
    """
    Reads a LAMMPS data file and returns the no of atoms, no of types, masses 
    of each types, and an Nx5 array containing ids, type, x, y, z coords
    """
    nlines = sum(1 for line in open(filename,'r'))

    boxvec = np.zeros((3,2))
    with open(filename,'r') as fid:
        for i in range(nlines):
            tmp = fid.readline().strip().split()
            if len(tmp) > 1 and tmp[1] == 'atoms':
                natoms = int(tmp[0])
            if len(tmp) > 2 and tmp[1] == 'atom' and tmp[2] == 'types':
                ntypes = int(tmp[0])
                masses = np.zeros(ntypes)
            if len(tmp) > 3 and tmp[-1] == 'xhi':
                boxvec[0,:] = tmp[0:2]
                boxvec[1,:] = fid.readline().strip().split()[0:2]
                boxvec[2,:] = fid.readline().strip().split()[0:2]
            if len(tmp) > 0 and tmp[0] == 'Masses':
                fid.readline()
                for j in range(ntypes):
                    masses[j] = fid.readline().strip().split()[1]
            if len(tmp) > 0 and tmp[0] == 'Atoms':
                fid.readline()
                tmp = fid.readline().strip().split()
                break
            
        ncol = len(tmp)
        pos = np.zeros((natoms,ncol))
        pos[0,:] = tmp
        for i in range(1,natoms):
            pos[i,:] = fid.readline().strip().split()
      
    return [natoms, ntypes, masses, pos]

#############################################################
def getMasses(ids,pos,masses):
    """
    This function gets the masses for all atoms in this term of the sum;
    useful when the unit cell isn't truly periodic, e.g. in a random alloy
    """
    nids = len(ids)
    massarr = np.zeros(nids)
    for i in range(nids):
        massarr[i] = masses[int(pos[ids[i],1]-1)] #look up mass for 
        #each atom
        
    return massarr

############################################################
def makeKpoints(prim,specialk,dk):
    """
    This function takes primitive lattice vectors, prim, and constructs
    reciprocal lattice vectors from them. It also takes an arbitrarily long
    (n,3) array of normalized special k points and an integer, dk.
    Reciprocal lattice vectors are constructed from the primitive lattice 
    vector and then those reciprocal lattice vectors are projected onto the
    special k points. An array of k points containing the special k points
    is then created and dk points between each k point are populated linearly.

    This function returns 2 variables: kpoints and kdist. kpoints is the array
    containing ((n-1)*dk,3) k points. kdist is cumulative sum of the euclidean
    k-space distance between each special k point - this is used later for
    plotting.
    """
    kvec = np.zeros((3,3)) #k space vectors 
    vol = np.dot(prim[0,:],np.cross(prim[1,:],prim[2,:])) #real space cell volume
    kvec[0,:] = 2*np.pi*np.cross(prim[1,:],prim[2,:])/vol
    kvec[1,:] = 2*np.pi*np.cross(prim[2,:],prim[0,:])/vol
    kvec[2,:] = 2*np.pi*np.cross(prim[0,:],prim[1,:])/vol
    #reciprocal lattice vectors with cubic symmetry are parallel to real 
    #space vectors
    for i in range(len(specialk)): #project into reciprocal space 
        specialk[i,:] = np.dot(kvec,specialk[i,:])
        
    nk = (len(specialk)-1)*dk
    kpoints = np.zeros((nk,3)) 
    for i in range(len(specialk)-1): #populate space between special k points
        for j in range(3): #kx, ky, kz
            kpoints[i*dk:(i+1)*dk,j] = np.linspace(specialk[i,j],
                    specialk[(i+1),j],dk) 
            
    kdist = np.zeros((len(specialk)))
    for i in range(len(specialk)-1): #euclidean distance between k points
        kdist[(i+1)] = np.sqrt((specialk[(i+1),0]-specialk[i,0])**2+
             (specialk[(i+1),1]-specialk[i,1])**2+
             (specialk[(i+1),2]-specialk[i,2])**2) 
    kdist = np.cumsum(kdist) #cumulative distance between special k points
    
    return [kpoints, kdist, len(kpoints[:,0])]

##########################################################
def makeTriclinic(n1,n2,n3,lammps='no',element='si'):
    """
    See docstring for makeFCCdiamond
    """
   
    if element == 'si':
        a = 5.431 #Si lattice constant
        masses = np.array([28.0855]) #atomic mass of Si
    if element == 'ge':
        a = 5.658 #Ge lattice constant
        masses = np.array([72.6400]) #atomic mass of Ge
    if element == 'c':
        a = 3.57 #C lattice constant
        masses = np.array([12.0107]) #atomic mass of C
    if (n1%2.0 != 0) and (n2%2.0 != 0) and (n3%2.0 != 0):
        sys.exit('Number of unit cells in each direction must be an even'
                 ' integer')
        
    nT = np.array([n1,n2,n3]).astype(int) #times to translate in each direction
    unit = np.array(([1,0,0,0], #FCC-diamond unit cell (basis-index,x,y,z,uc)
                     [2,1,1,1])).astype(float)
    
    uc = np.array([0,0])
    
    tmp = cp.deepcopy(unit)
    pos = cp.deepcopy(unit) #basis-index and coordinates of atoms
    tuc = cp.deepcopy(uc)
    for i in range(nT[0]-1): #translate in a1-direction 
        tmp[:,1] = tmp[:,1]+2
        pos = np.append(pos,tmp,axis=0)
        tuc = tuc+1
        uc = np.append(uc,tuc)
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    for i in range(nT[1]-1): #translate in a2-direction
        tmp[:,1] = tmp[:,1]+1
        tmp[:,2] = tmp[:,2]+3
        pos = np.append(pos,tmp,axis=0)
        tuc2 = tuc+max(uc)+1
        uc = np.append(uc,tuc2)
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    for i in range(nT[2]-1): #translate in a3-direction
        tmp[:,1] = tmp[:,1]+1
        tmp[:,2] = tmp[:,2]+1
        tmp[:,3] = tmp[:,3]+4
        pos = np.append(pos,tmp,axis=0)
        tuc2 = tuc+max(uc)+1
        uc = np.append(uc,tuc2)
    
    num = len(pos[:,0])
    ids = np.zeros((num,1))
    ids[:,0] = np.arange(0,num,1)+1 #will be used later    
    pos = np.append(ids,pos,axis=1)
    pos[:,2] = pos[:,2]*a*np.sqrt(2.0)/4.0 #rescale coordinates
    pos[:,3] = pos[:,3]*a/np.sqrt(24.0) #rescale coordinates
    pos[:,4] = pos[:,4]*a/np.sqrt(3.0)/4.0 #rescale coordinates
    ids = cp.deepcopy(pos[:,1])
    pos[:,1] = 1
    
    if lammps != 'no':
        with open(lammps, 'w') as fid:
            xbuff = (pos[np.argwhere(pos[:,3] == 0)[:,0],2][1]-
                     pos[np.argwhere(pos[:,3] == 0)[:,0],2][0])/2.0
            xmax = pos[np.argwhere(pos[:,3] == 0)[:,0],2].max()+xbuff
            xmin = 0-xbuff #by construction
            ybuff = (np.unique(pos[np.argwhere(pos[:,4] == 0)[:,0],3])[1]-
                     np.unique(pos[np.argwhere(pos[:,4] == 0)[:,0],3])[0])/2.0
            ymax = pos[np.argwhere(pos[:,4] == 0)[:,0],3].max()+ybuff
            ymin = 0-ybuff #by construction
            zbuff = (np.unique(pos[:,4])[2]-np.unique(pos[:,4])[1])/2.0
            zmax = pos[:,4].max()+zbuff
            zmin = 0-zbuff #by construction
            
            xy = pos[np.argwhere(pos[:,4] == pos[:,4].max())[:,0],:]
            xy = xy[np.argwhere(xy[:,3] == xy[:,3].min())[:,0],2].min()
            xz = xy
            yz = pos[np.argwhere(pos[:,4] == pos[:,4].max())[:,0],3].min()
    
            fid.write(str('LAMMPS DATA FILE\n'))
        
            fid.write('\n' + str(len(pos)) + ' atoms\n')
            fid.write('\n' + str(len(masses)) + ' atom types\n')
            fid.write('\n' + str(xmin)+' '+str(xmax)+' xlo'+' xhi\n')
            fid.write(str(ymin)+' '+str(ymax)+' ylo'+' yhi\n')
            fid.write(str(zmin)+' '+str(zmax)+' zlo'+' zhi\n')
            fid.write(str(xy)+' '+str(xz)+' '+
                          str(yz)+' xy xz yz\n')
            fid.write('\nMasses\n')
            fid.write('\n' + str(1) + ' ' + str(float(masses)))
            fid.write('\n\nAtoms\n\n')
            for i in range(len(pos)-1):
                fid.write(str(int(i+1)) + ' ' + str(int(pos[i,1])) + ' ' 
                          + str(pos[i,2]) + ' ' +
                        str(pos[i,3]) + ' ' + str(pos[i,4]) + '\n')
            fid.write(str(len(pos)) +  ' ' + str(int(pos[-1,1])) + ' ' 
                      + str(pos[-1,2]) + ' ' +
                    str(pos[-1,3]) + ' ' + str(pos[-1,4]))
            
    return [num, pos, masses, uc, ids, a] 

########################################################
def makeSL(nx,ny,nz,period,lammps='no',element='si/ge'):
    """
    This function replicates a single Si/Ge superlattice period as a 
    supercell for SED calculations.
    5 arguments: nx, ny, nz are the number of times to replicate the supercell
    in x, y, and z respectively. The next argument, period, is the TOTAL number
    or unit cells in the period. 2*N_Si = 2*N_Ge = period. 
    This function returns 5 argumnets: num is the number of atoms. pos is the
    array containing atom ids, types, and x, y, z coords. masses are the masses
    of the two atoms in the array, i.e. Si/Ge. ids are the ids of each atom 
    within each unit cell. uc are the unit cell ids corresponing to each atom
    in pos and ids. a is the lattice constant =  a_si/2.0+a_ge/2.0 = avg of Si
    and Ge lattice constants. Optional argument 'lammps' is default 'no'. 
    If you want to write a lammps data file, change lammps to the file name.
    """
    #total number of unit cells in period; N_Si = N_Ge
    if period %2.0 !=0:
        sys.exit('Period must be even integer')
    nSi = int(period/2)
        
    if element == 'si/ge':
        a = 5.431/2.0+5.658/2.0
        masses = np.array([28.0855,72.6400])
    elif element == 'si':
        a = 5.431
        masses = np.array([28.0855,28.0855])
    elif element == 'ge':
        a = 5.658
        masses = np.array([72.6400,72.6400])
    
    basis = np.array([[0,0,0], 
                      [0,2,2],
                      [2,0,2],
                      [2,2,0],
                      [1,1,1],
                      [3,3,1],
                      [1,3,3],
                      [3,1,3]]).astype(float) #8 atom conventional 
                                              #FCC-diamond cell
    
    types = np.array([1,1,1,1,1,1,1,1]).reshape(8,1)
    basis = np.append(types,basis,axis=1)
    uc = np.zeros((8*period))
    ids = np.arange(0,8*period)
    
    tmp = cp.deepcopy(basis)
    pos = cp.deepcopy(basis)
    for i in range(nSi-1): #replicate Si slab to create period
        tmp[:,1] = tmp[:,1]+4
        pos = np.append(pos,tmp,axis=0)
    tmp = cp.deepcopy(pos)
    tmp[:,0] = 2    
    tmp[:,1] = tmp[:,1]+4*nSi
    
    pos = np.append(pos,tmp,axis=0) #add Ge slab to Si slab
    
    #replicate in x, y, z
    tmp = cp.deepcopy(pos)
    xmax = tmp[:,1].max()
    tuc = cp.deepcopy(uc)
    tids = cp.deepcopy(ids)
    for i in range(nx-1): #x
        tmp[:,1] = tmp[:,1]+xmax+1
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+1
        uc = np.append(uc,tuc[:])
        ids = np.append(ids,tids)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    ucmax = uc.max()
    tids = cp.deepcopy(ids)
    for i in range(ny-1): #y
        tmp[:,2] = tmp[:,2]+4
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+ucmax+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tids)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    ucmax = uc.max()
    tids = cp.deepcopy(ids)
    for i in range(nz-1): #z
        tmp[:,3] = tmp[:,3]+4
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+ucmax+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tids)    
    
    num = len(pos[:,0])
    tmp = np.arange(1,num+1).reshape(num,1)
    pos = np.append(tmp,pos,axis=1)
    pos[:,2] = pos[:,2]*a/4.0
    pos[:,3] = pos[:,3]*a/4.0
    pos[:,4] = pos[:,4]*a/4.0
    
    if lammps != 'no':
        with open(lammps, 'w') as fid:
                buff = a/8.0
                xmax = pos[:,2].max()+buff
                xmin = 0-buff
                ymax = pos[:,3].max()+buff
                ymin = 0-buff
                zmax = pos[:,4].max()+buff
                zmin = 0-buff 
        
                fid.write(str('LAMMPS pos FILE\n'))
            
                fid.write('\n' + str(num) + ' atoms\n')
                fid.write('\n' + str(len(masses)) + ' atom types\n')
                fid.write('\n' + str(xmin)+' '+str(xmax)+' xlo'+' xhi\n')
                fid.write(str(ymin)+' '+str(ymax)+' ylo'+' yhi\n')
                fid.write(str(zmin)+' '+str(zmax)+' zlo'+' zhi\n')
                fid.write('\nMasses\n')
                for i in range(len(masses)):
                    fid.write('\n' + str(i+1) + ' ' + str(float(masses[i])))
                fid.write('\n\nAtoms\n\n')
                for i in range(num-1):
                    fid.write(str(int(i+1)) + ' ' + str(int(pos[i,1])) + ' ' 
                              + str(pos[i,2]) + ' ' +
                            str(pos[i,3]) + ' ' + str(pos[i,4]) + '\n')
                fid.write(str(len(pos)) +  ' ' + str(int(pos[-1,1])) + ' ' 
                          + str(pos[-1,2]) + ' ' +
                        str(pos[-1,3]) + ' ' + str(pos[-1,4]))
    return [num, pos, masses, uc, ids, a]

##########################################################
def makeGaN(nx,ny,nz,lammps='no'):
    """
    Same as makeSL except return a & c, lattice constants for x=y and 
    z respectively
    """
    masses = np.array([69.723,14.007]) 
    a = 3.189 #a = b = diagonal length
    c = np.round(np.sqrt(8/3.0)*a,decimals=3) #5.185/2.0 #ideal c = sqrt(8/3)*a
    
    basis = np.array(([1,0,0,0], #Ga
                      [2,0,0,3], #N
                      [1,2,0,4], #Ga
                      [2,2,0,7], #N
                      [1,3,1,0], #Ga
                      [2,3,1,3], #N
                      [1,5,1,4], #Ga
                      [2,5,1,7])).astype(float) #N
    uc = np.array([0,0,0,0,0,0,0,0])
    ids = np.arange(0,8)
    
    pos = cp.deepcopy(basis)
    tmp = cp.deepcopy(basis)
    tuc = cp.deepcopy(uc)
    tmpids = cp.deepcopy(ids)
    for i in range(nx-1):
        tmp[:,1] = tmp[:,1]+(6)
        pos = np.append(pos,tmp,axis=0)
        tuc = tuc+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tmpids,axis=0)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    tmpids = cp.deepcopy(ids)
    for i in range(ny-1):
        tmp[:,2] = tmp[:,2]+2
        tuc = tuc+nx
        uc = np.append(uc,tuc)
        pos = np.append(pos,tmp,axis=0)
        ids = np.append(ids,tmpids,axis=0)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    tmpids = cp.deepcopy(ids)
    for i in range(nz-1):
        tmp[:,3] = tmp[:,3]+8
        pos = np.append(pos,tmp,axis=0)
        tuc = tuc+nx*ny
        uc = np.append(uc,tuc)
        ids = np.append(ids,tmpids,axis=0)
        
    pos[:,1] = pos[:,1]*np.sqrt(3)*a/6.0
    pos[:,1] = np.round(pos[:,1],decimals=4)
    pos[:,2] = pos[:,2]*a/2.0
    pos[:,2] = np.round(pos[:,2],decimals=4)
    pos[:,3] = pos[:,3]*c/8.0
    pos[:,3] = np.round(pos[:,3],decimals=4)
    
    num = len(pos)
    pos = np.append(np.arange(1,num+1).reshape(num,1),pos,axis=1)
    
    if lammps != 'no':    
        with open(lammps,'w') as fid:
            xbuff = 0.4603
            ybuff = 0.79725
            zbuff = 0.3255
            fid.write('LAMMPS DATA FILE FOR SED\n')
            fid.write('\n'+str(num)+' atoms\n')
            fid.write('\n'+str(len(masses))+' atom types\n')
            fid.write('\n'+str(pos[:,2].min()-xbuff)+' '+
                      str(pos[:,2].max()+xbuff)+' xlo xhi\n')
            fid.write(str(pos[:,3].min()-ybuff)+' '+str(pos[:,3].max()+ybuff)+
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

    return [num, pos, masses, uc, ids, a, c]    

########################################################
def makeInGaN(nx,ny,nz,x=0,lammps='no'):
    """
    Makes In_xGa_(1-x)N Alloy
    Set x = 0 for GaN
    Same as makeSL except return a & c, lattice constants for x=y and 
    z respectively
    """

    if x < 0 or x >= 1: 
        sys.exit('\tUSAGE ERROR: x, concentration of In defects, must be in \n'
                 '\tinterval [0,1)')
    if x == 0:
        masses = np.array([69.723,14.007])
    else:
        masses = np.array([69.723,14.007,114.818]) 
    a = 3.189 #a = b = diagonal length
    c = np.round(np.sqrt(8/3.0)*a,decimals=3) #5.185/2.0 #ideal c = sqrt(8/3)*a
    
    basis = np.array(([1,0,0,0], #Ga
                      [2,0,0,3], #N
                      [1,2,0,4], #Ga
                      [2,2,0,7], #N
                      [1,3,1,0], #Ga
                      [2,3,1,3], #N
                      [1,5,1,4], #Ga
                      [2,5,1,7])).astype(float) #N
    uc = np.array([0,0,0,0,0,0,0,0])
    ids = np.arange(0,8)
    
    pos = cp.deepcopy(basis)
    tmp = cp.deepcopy(basis)
    tuc = cp.deepcopy(uc)
    tmpids = cp.deepcopy(ids)
    for i in range(nx-1):
        tmp[:,1] = tmp[:,1]+(6)
        pos = np.append(pos,tmp,axis=0)
        tuc = tuc+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tmpids,axis=0)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    tmpids = cp.deepcopy(ids)
    for i in range(ny-1):
        tmp[:,2] = tmp[:,2]+2
        tuc = tuc+nx
        uc = np.append(uc,tuc)
        pos = np.append(pos,tmp,axis=0)
        ids = np.append(ids,tmpids,axis=0)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    tmpids = cp.deepcopy(ids)
    for i in range(nz-1):
        tmp[:,3] = tmp[:,3]+8
        pos = np.append(pos,tmp,axis=0)
        tuc = tuc+nx*ny
        uc = np.append(uc,tuc)
        ids = np.append(ids,tmpids,axis=0)
        
    pos[:,1] = pos[:,1]*np.sqrt(3)*a/6.0
    pos[:,1] = np.round(pos[:,1],decimals=4)
    pos[:,2] = pos[:,2]*a/2.0
    pos[:,2] = np.round(pos[:,2],decimals=4)
    pos[:,3] = pos[:,3]*c/8.0
    pos[:,3] = np.round(pos[:,3],decimals=4)
    
    num = len(pos)
    pos = np.append(np.arange(1,num+1).reshape(num,1),pos,axis=1)
    
    if x != 0:
        ganids = np.argwhere(pos[:,1] == 1)
        ngan = len(ganids)
        nin = int(x*ngan)
        np.random.shuffle(ganids)
        inids = ganids[0:nin]
        pos[inids,1] = 3
    
    if lammps != 'no':    
        with open(lammps,'w') as fid:
            xbuff = 0.4603
            ybuff = 0.79725
            zbuff = 0.3255
            fid.write('LAMMPS DATA FILE FOR SED\n')
            fid.write('\n'+str(num)+' atoms\n')
            fid.write('\n'+str(len(masses))+' atom types\n')
            fid.write('\n'+str(pos[:,2].min()-xbuff)+' '+
                      str(pos[:,2].max()+xbuff)+' xlo xhi\n')
            fid.write(str(pos[:,3].min()-ybuff)+' '+str(pos[:,3].max()+ybuff)+
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

    return [num, pos, masses, uc, ids, a, c]    
########################################################
def makeSi(nx,ny,nz,lammps='no'):
    """
    This function replicates a single Si/Ge superlattice period as a 
    supercell for SED calculations.
    5 arguments: nx, ny, nz are the number of times to replicate the supercell
    in x, y, and z respectively. The next argument, period, is the TOTAL number
    or unit cells in the period. 2*N_Si = 2*N_Ge = period. 
    This function returns 5 argumnets: num is the number of atoms. pos is the
    array containing atom ids, types, and x, y, z coords. masses are the masses
    of the two atoms in the array, i.e. Si/Ge. ids are the ids of each atom 
    within each unit cell. uc are the unit cell ids corresponing to each atom
    in pos and ids. a is the lattice constant =  a_si/2.0+a_ge/2.0 = avg of Si
    and Ge lattice constants. Optional argument 'lammps' is default 'no'. 
    If you want to write a lammps data file, change lammps to the file name.
    """
    a = 5.431
    mass = 28.0855
    
    basis = np.array([[0,0,0], 
                      [0,2,2],
                      [2,0,2],
                      [2,2,0],
                      [1,1,1],
                      [3,3,1],
                      [1,3,3],
                      [3,1,3]]).astype(float) #8 atom conventional 
                                              #FCC-diamond cell
    
    types = np.array([1,1,1,1,1,1,1,1]).reshape(8,1)
    basis = np.append(types,basis,axis=1)
    uc = np.zeros(8)
    ids = np.arange(0,8)
    
    #replicate in x, y, z
    pos = cp.deepcopy(basis)
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    tids = cp.deepcopy(ids)
    for i in range(nx-1): #x
        tmp[:,1] = tmp[:,1]+4
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+1
        uc = np.append(uc,tuc[:])
        ids = np.append(ids,tids)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    ucmax = uc.max()
    tids = cp.deepcopy(ids)
    for i in range(ny-1): #y
        tmp[:,2] = tmp[:,2]+4
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+ucmax+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tids)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    ucmax = uc.max()
    tids = cp.deepcopy(ids)
    for i in range(nz-1): #z
        tmp[:,3] = tmp[:,3]+4
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+ucmax+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tids)    
    
    num = len(pos[:,0])
    tmp = np.arange(1,num+1).reshape(num,1)
    pos = np.append(tmp,pos,axis=1)
    pos[:,2] = pos[:,2]*a/4.0
    pos[:,3] = pos[:,3]*a/4.0
    pos[:,4] = pos[:,4]*a/4.0
    
    if lammps != 'no':
        with open(lammps, 'w') as fid:
                buff = a/8.0
                xmax = pos[:,2].max()+buff
                xmin = 0-buff
                ymax = pos[:,3].max()+buff
                ymin = 0-buff
                zmax = pos[:,4].max()+buff
                zmin = 0-buff 
        
                fid.write(str('LAMMPS pos FILE\n'))
            
                fid.write('\n' + str(num) + ' atoms\n')
                fid.write('\n1 atom types\n')
                fid.write('\n' + str(xmin)+' '+str(xmax)+' xlo'+' xhi\n')
                fid.write(str(ymin)+' '+str(ymax)+' ylo'+' yhi\n')
                fid.write(str(zmin)+' '+str(zmax)+' zlo'+' zhi\n')
                fid.write('\nMasses\n')
                fid.write('\n1 ' + str(mass))
                fid.write('\n\nAtoms\n\n')
                for i in range(num-1):
                    fid.write(str(int(i+1)) + ' ' + str(int(pos[i,1])) + ' ' 
                              + str(pos[i,2]) + ' ' +
                            str(pos[i,3]) + ' ' + str(pos[i,4]) + '\n')
                fid.write(str(len(pos)) +  ' ' + str(int(pos[-1,1])) + ' ' 
                          + str(pos[-1,2]) + ' ' +
                        str(pos[-1,3]) + ' ' + str(pos[-1,4]))
    return [num, pos, mass, uc, ids, a]


#############################################################
def makeAlloy(nx,ny,nz,ux=1,uy=1,uz=1,x=0.0,lammps='no'):
    """
    """
    a = 5.431
    
    if x < 0 or x >= 1: 
        sys.exit('\tUSAGE ERROR: x, concentration of In defects, must be in \n'
                 '\tinterval [0,1)')
    if x == 0:
        masses = np.array([28.0855])
    else:
        masses = np.array([28.0855,72.64]) 
    
    basis = np.array([[0,0,0], 
                      [0,2,2],
                      [2,0,2],
                      [2,2,0],
                      [1,1,1],
                      [3,3,1],
                      [1,3,3],
                      [3,1,3]]).astype(float) #8 atom conventional 
                                              #FCC-diamond cell
    #replicate basis to create super cell
    pos = cp.deepcopy(basis)
    tmp = cp.deepcopy(pos)
    for i in range(ux-1): #x
        tmp[:,0] = tmp[:,0]+4
        pos = np.append(pos,tmp,axis=0)
    tmp = cp.deepcopy(pos)
    for i in range(uy-1): #y
        tmp[:,1] = tmp[:,1]+4
        pos = np.append(pos,tmp,axis=0)
    tmp = cp.deepcopy(pos)
    for i in range(uz-1): #z
        tmp[:,2] = tmp[:,2]+4
        pos = np.append(pos,tmp,axis=0)
        
    nb = len(pos[:,0])
    types = np.ones((nb,1))
    
    if x != 0:
        ids = np.arange(nb)
        nge = int(x*nb)
        np.random.shuffle(ids)
        geids = ids[:nge]
        types[geids,0] = 2
        
    basis = np.append(types.reshape(nb,1),pos,axis=1)
    uc = np.zeros(nb)
    ids = np.arange(0,nb)
    
    ##replicate in x, y, z
    pos = cp.deepcopy(basis)
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    tids = cp.deepcopy(ids)
    dx = int(pos[:,1].max())+1
    for i in range(nx-1): #x
        tmp[:,1] = tmp[:,1]+dx
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+1
        uc = np.append(uc,tuc[:])
        ids = np.append(ids,tids)
    
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    ucmax = uc.max()
    tids = cp.deepcopy(ids)
    dy = int(pos[:,2].max())+1
    for i in range(ny-1): #y
        tmp[:,2] = tmp[:,2]+dy
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+ucmax+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tids)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    ucmax = uc.max()
    tids = cp.deepcopy(ids)
    dz = int(pos[:,3].max())+1
    for i in range(nz-1): #z
        tmp[:,3] = tmp[:,3]+dz
        pos = np.append(pos,tmp,axis=0)
        tuc[:] = tuc[:]+ucmax+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tids)    
    
    num = len(pos[:,0])
    tmp = np.arange(1,num+1).reshape(num,1)
    pos = np.append(tmp,pos,axis=1)
    pos[:,2] = pos[:,2]*a/4.0
    pos[:,3] = pos[:,3]*a/4.0
    pos[:,4] = pos[:,4]*a/4.0
    
    if lammps != 'no':    
        with open(lammps,'w') as fid:
            buff = a/8
            fid.write('LAMMPS DATA FILE FOR SED\n')
            fid.write('\n'+str(num)+' atoms\n')
            fid.write('\n'+str(len(masses))+' atom types\n')
            fid.write('\n'+str(pos[:,2].min()-buff)+' '+
                      str(pos[:,2].max()+buff)+' xlo xhi\n')
            fid.write(str(pos[:,3].min()-buff)+' '+str(pos[:,3].max()+buff)+
                      ' ylo yhi\n')
            fid.write(str(pos[:,4].min()-buff)+' '+str(pos[:,4].max()+buff)+
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

############################################################
def makehBN(nx,ny,ux=1,uy=1,x=0.0,iso='N',lammps='none'):
    """
    Makes a 2D hBN monolayer. The minimum unit cell is 4 atom ortho.
    Define a larger supercell using the ux and uy and the structure using
    nx and ny. LAMMPS = 'none' for no data file or the name of the data file.
    
    This function finds the bonds, angles, and impropers and writes them to
    the LAMMPS data file; to use the potential in : 
    "J. Phys. Chem. Lett. 2018, 9, 1584−1591" these are needed.
    """
    if x < 0 or x >= 1:
        sys.exit('\tUSAGE ERROR: x, concentration of defects, must be in '
                 'interval [0,1)')
        
    if x > 0:
        if iso == 'N':
            masses = np.array([11.009,14.003,15.000]) #masses in AMU
        elif iso == 'B':
            masses = np.array([11.009,14.003,10.013]) #masses in AMU
        else:
            sys.exit('\tUSAGE ERROR: enter either \'N\' or \'B\' for '
                     'isotope type.')
    else:
        masses = np.array([11.009,14.003]) #masses in AMU
        
    ## ORTHO UNIT CELL LATTICE VECTORS
    ax = 2.510 #2*1.255
    ay = 4.348 # 6*0.724574
    
    ## 4 ATOM CONVENTIONAL CELL -- BASIS
    basis = np.array([[1,   1,    0.907,   0,   0,   0],   #B
                      [1,   2,   -0.907,   1,   1,   0],   #N
                      [1,   1,    0.907,   1,   3,   0],   #B
                      [1,   2,   -0.907,   0,   4,   0]])  #N
                  #mol.id  #type  #charge   #x   #y   #z
    
    ## REPLICATE TO FORM SUPERCELL
    pos = cp.deepcopy(basis)
    tmp = cp.deepcopy(basis)
    for i in range(ux-1):
        tmp[:,3] = tmp[:,3]+2
        pos = np.append(pos,tmp,axis=0)
    tmp = cp.deepcopy(pos)
    for i in range(uy-1):
        tmp[:,4] = tmp[:,4]+6
        pos = np.append(pos,tmp,axis=0)
    
    nb = len(pos[:,0])
    uc = np.zeros(nb)
    ids = np.arange(0,nb)
    
    ## GENERATE DEFECTS
    if x != 0:
        if iso == 'N':
            defids = np.argwhere(pos[:,1] == 2)
            ndef = int(x*nb)
            np.random.shuffle(defids)
            defids = defids[:ndef]
            pos[defids,1] = 3
        else:
            defids = np.argwhere(pos[:,1] == 1)
            ndef = int(x*nb)
            np.random.shuffle(defids)
            defids = defids[:ndef]
            pos[defids,1] = 3
                    
    ## REPLICATE TO FORM STRUCTURE
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    tids = cp.deepcopy(ids)
    dx = pos[:,3].max()+1
    for i in range(nx-1):
        tmp[:,3] = tmp[:,3]+dx
        pos = np.append(pos,tmp,axis=0)
        tuc = tuc+1
        uc = np.append(uc,tuc)
        ids = np.append(ids,tids)
        
    tmp = cp.deepcopy(pos)
    tuc = cp.deepcopy(uc)
    duc = uc.max()+1
    tids = cp.deepcopy(ids)
    dy = pos[:,4].max()+2
    for i in range(ny-1):
        tmp[:,4] = tmp[:,4]+dy
        pos = np.append(pos,tmp,axis=0)
        tuc = tuc+duc
        uc = np.append(uc,tuc)
        ids = np.append(ids,tids)
        
    ## ORGANIZE
    num = len(pos[:,0])
    pos = np.append(np.arange(1,num+1,1).reshape(num,1),pos,axis=1) #ids
    pos = pos.astype(float)
    pos[:,4] = pos[:,4]*ax/2 
    pos[:,5] = pos[:,5]*ay/6
    
    ## DETERMINE BOND INFO
    bx = ax/4 #buffer between periodic images
    by = ay/6 # '' ''
    xmin = pos[:,4].min()-bx
    xmax = pos[:,4].max()+bx
    ymin = pos[:,5].min()-by
    ymax = pos[:,5].max()+by
    zmin = pos[:,6].min()-100
    zmax = pos[:,6].max()+100
        
    lx = (xmax-xmin)/2 #distance to shift for pbc
    ly = (ymax-ymin)/2
    
    nl = np.zeros((num,4))
    nd = np.zeros((num,4))
    print('\n\tDetermining bonds, angles, and impropers.' 
          ' This will take a while ...\n')
    for i in range(num): #loop over particles
        if i != 0 and i%(num//10) == 0:
            print('\t\tNow '+str(np.round(10*i/(num//10),0))+
                  ' % done.')
            
        rvec = pos[i,4:6]
        nnids = np.intersect1d(np.intersect1d(np.argwhere(pos[:,4] <= rvec[0]+1.5),
                               np.argwhere(pos[:,4] >= rvec[0]-1.5)),
                               np.intersect1d(np.argwhere(pos[:,5] <= rvec[1]+1.5),
                               np.argwhere(pos[:,5] >= rvec[1]-1.5)))
        nnids = np.append(nnids,np.argwhere(pos[:,4] <= xmin+1.5))
        nnids = np.append(nnids,np.argwhere(pos[:,4] >= xmax-1.5))
        nnids = np.append(nnids,np.argwhere(pos[:,5] <= ymin+1.5))
        nnids = np.append(nnids,np.argwhere(pos[:,5] >= ymax-1.5))
        nnids = np.unique(nnids)
        
        nn = len(nnids)
        tmp = np.zeros((2,nn)) # ids, distance
        for j in range(nn): #loop over neighbors
            rx = pos[nnids[j],4]-rvec[0] # vector from partice i to j
            ry = pos[nnids[j],5]-rvec[1] # vector from partice i to j
    
            ## Minimum image convention
            if rx >= lx:
                rx = rx-lx*2
            elif rx <= -lx:
                rx = rx+lx*2
            if ry >= ly:
                ry = ry-ly*2
            elif ry <= -ly:
                ry = ry+ly*2
            
            dist = np.sqrt(rx**2+ry**2)
            tmp[0,j] = pos[nnids[j],0]-1 #index
            tmp[1,j] = dist #distance
        
        nl[i,:] = tmp[0,np.argsort(tmp[1,:])[0:4]] #only keep the first few. 
        nd[i,:] = tmp[1,np.argsort(tmp[1,:])[0:4]] #saves memory.
    print('\n\tDone determining bonds, angles, and impropers!')
    
    ## BONDS
    nbonds = num*3
    bonds = np.zeros((nbonds,4))
    d = 0
    for i in range(num):
        for j in range(3):
            bonds[d,2] = i+1
            bonds[d,3] = nl[i,j+1]+1
            if (len(np.argwhere(bonds[:,2] == bonds[d,3])) != 0 and
                len(np.argwhere(bonds[:,3] == bonds[d,2])) != 0):
                    bonds[d,0] = 1
            d = d+1
    
    bonds = np.delete(bonds,np.argwhere(bonds[:,0] == 1),axis=0)    
    nbonds = len(bonds[:,0])
    bonds[:,0] = np.arange(1,nbonds+1)
    bonds[:,1] = 1
            
    ## ANGLES
    nangles = num*3
    angles = np.zeros((nangles,5))
    angles[:,0] = np.arange(1,nangles+1)
    d = 0
    for i in range(num):
        if pos[int(nl[i,0]),2] == 1:
            atype = 1
        elif pos[int(nl[i,0]),2] == 2:
            atype = 2
        else:
            if iso == 'B':
                atype = 1
            else:
                atype == 2
        for j in range(3):
            angles[d,1] = atype
            angles[d,3] = i+1
            if j == 0:
                angles[d,2] = nl[i,j+1]+1
                angles[d,4] = nl[i,j+2]+1
            if j == 1:
                angles[d,2] = nl[i,j+1]+1
                angles[d,4] = nl[i,j+2]+1
            if j == 2:
                angles[d,2] = nl[i,j+1]+1
                angles[d,4] = nl[i,j-1]+1
            d = d+1
            
    ## IMPROPERS
    nimprop = num
    improp = np.zeros((num,6))
    improp[:,0] = np.arange(1,nimprop+1)
    for i in range(num):
        if pos[int(nl[i,0]),2] == 1:
            atype = 1
        elif pos[int(nl[i,0]),2] == 2:
            atype = 2
        else:
            if iso == 'B':
                atype = 1
            else:
                atype == 2
        improp[i,1] = atype
        improp[i,2:6] = nl[i,0:4]+1
            
    ## WRITE LAMMPS
    if lammps != 'none':    
        with open(lammps,'w') as fid:
            fid.write('LAMMPS hBN\n')
            fid.write('\n'+str(num)+' atoms')
        
            fid.write('\n'+str(nbonds)+' bonds')
            fid.write('\n'+str(nangles)+' angles')
            fid.write('\n'+str(nimprop)+' impropers\n')
            
            fid.write('\n'+str(len(masses))+' atom types')
            fid.write('\n'+str(1)+' bond types')
            fid.write('\n'+str(2)+' angle types')
            fid.write('\n'+str(2)+' improper types\n')
            
            fid.write('\n'+str(xmin)+' '+str(xmax)+' xlo xhi\n')
            fid.write(str(ymin)+' '+str(ymax)+' ylo yhi\n')
            fid.write(str(zmin)+' '+str(zmax)+' zlo zhi\n')
            fid.write('\nMasses\n\n')
            
            for i in range(len(masses)):
                fid.write(str(i+1)+' '+str(masses[i])+'\n')
                
            fid.write('\nAtoms\n\n')
            for i in range(num):
                fid.write(str(int(pos[i,0]))+' '+str(int(pos[i,1]))+' '
                          +str(int(pos[i,2]))+' '+str(pos[i,3])+' '+
                          str(pos[i,4])+' '+str(pos[i,5])+' '+str(pos[i,6])+'\n')
            
            bonds = bonds.astype(int)
            fid.write('\nBonds\n\n')
            for i in range(nbonds):
                fid.write(str(bonds[i,0])+' '+str(bonds[i,1])+' '+str(bonds[i,2])+
                          ' '+str(bonds[i,3])+'\n')
                
            angles = angles.astype(int)
            fid.write('\nAngles\n\n')
            for i in range(nangles):
                fid.write(str(angles[i,0])+' '+str(angles[i,1])+' '+str(angles[i,2])+
                          ' '+str(angles[i,3])+' '+str(angles[i,4])+'\n')
                
            improp = improp.astype(int)
            fid.write('\nImpropers\n\n')
            for i in range(nimprop-1):
                fid.write(str(improp[i,0])+' '+str(improp[i,1])+' '+str(improp[i,2])+
                          ' '+str(improp[i,3])+' '+str(improp[i,4])+' '
                          +str(improp[i,5])+'\n')
            fid.write(str(improp[-1,0])+' '+str(improp[-1,1])+' '+str(improp[-1,2])+
                      ' '+str(improp[-1,3])+' '+str(improp[-1,4])+' '
                      +str(improp[-1,5]))
            
    return [num, pos, masses, uc, ids, ax, ay]
##########################################################
def writeSED(outfile,thz,kpoints,sed,dos):
    """
    This function is simple. It writes the frequency data array, k points, 
    SED matrix, and DOS array to file 'outfile'. Read file with readSED
    """
    nf = len(thz)
    nk = len(kpoints[:,0])
    sed = np.reshape(sed,(nf*nk,1))

    with open(outfile, 'w') as fid:
        fid.write('nf = '+str(nf)+'\n')
        fid.write('nk = '+str(nk)+'\n')
        for i in range(nf):
            fid.write(str(thz[i])+'\n')
        for i in range(nk):
            fid.write(str(kpoints[i,0])+'\t'+str(kpoints[i,1])+'\t'+
                      str(kpoints[i,2])+'\n')
        for i in range(nk*nf):
            fid.write(str(sed[i,0])+'\n')
        for i in range(nf):
            fid.write(str(dos[i,0])+'\n')
         
######################################################################
def readSED(infile):
    """
    This function reads in the SED outout file and returns THz, kpoints, SED, 
    and DOS written to file using writeSED
    """
    with open(infile,'r') as fid:
        nf = int(fid.readline().strip().split()[2])
        nk = int(fid.readline().strip().split()[2])
        thz = np.zeros((nf,1))
        kpoints = np.zeros((nk,3))
        sed = np.zeros((nf*nk,1)).astype(float)
        dos = np.zeros((nf,1))
        for i in range(nf):
            thz[i,0] = float(fid.readline())
        for i in range(nk):
            kpoints[i,:] = fid.readline().strip().split()[:]
        kpoints = kpoints.astype(float)
        for i in range(nf*nk):
            sed[i,0] = float(fid.readline())
        sed = np.reshape(sed,(nf,nk))
        for i in range(nf):
            dos[i,0] = float(fid.readline())
    return thz, kpoints, np.real(sed), dos

##########################################################            
def tic():
    """
    Same as MATLAB tic and toc functions. Use ty.tic() at the beginning of
    code you want to time and ty.toc() at the end. Once ty.toc() is reached,
    elapsted time will be printed to screen and optionally (by default) written
    to 'log.txt' file.
    """
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(logFlag='yes'):
    """
    Same as MATLAB tic and toc functions. Use ty.tic() at the beginning of
    code you want to time and ty.toc() at the end. Once ty.toc() is reached,
    elapsted time will be printed to screen and optionally (by default) written
    to 'log.txt' file.
    """
    import time
    if 'startTime_for_tictoc' in globals():
        if logFlag == 'yes':
            log("\n\tElapsed time is "+str(np.round(time.time()-
                           startTime_for_tictoc,decimals=3))+" seconds.")
        else:
            print(("\n\tElapsed time is "+
                  str(np.round(time.time()-
                           startTime_for_tictoc,decimals=3))+" seconds."))
    else:
        print("\n\t\tToc: start time not set") 
        
        
##########################################################     
def VDOS(dos,vels,tn,num,dt,dn,thz):
    """
    This function calculates vibrational density of states from EMD velocity
    data. Intended to be used adjunct to SED code. 
    """
    vels = vels.reshape(tn,num*3)
    dos = np.zeros((tn,num*3))
    velsfft = np.fft.fft(vels,axis=0)*dt*dn
    vdos = (np.multiply(abs(velsfft),abs(velsfft))/
               np.tile(np.multiply(vels,vels).mean(axis=0),(tn,1))/(tn*dt*dn))
    dos[:,0] = dos[:,0]+vdos.mean(axis=1) #total
    dos[:,1] = dos[:,1]+vdos[:,0::3].mean(axis=1) #x
    dos[:,2] = dos[:,2]+vdos[:,1::3].mean(axis=1) #y
    dos[:,3] = dos[:,3]+vdos[:,2::3].mean(axis=1) #z

    return dos

##########################################################
def log(string,outfile='log.txt',suppress='no',new='no'):
    """
    This function prints output to a file called log.txt and to the screen. 
    Useful for tracking whats happening when submitted using qsub or slurm etc.
    If you don't want to print to screen, enter supress='yes'
    """
    if new == 'no':
        with open(outfile,'a') as fid:
            fid.write(string+'\n')
    else:
        with open(outfile,'w') as fid:
            fid.write(string+'\n')
    if suppress == 'no':
        print(string)
    
def printParams(dt,dn,num,steps,split,nk,klabel,thz):
    """
    Writes input data to screen and file
    """
    log('\tMD timestep:\t\t'+str(dt*1e15)+'\tfs')
    log('\tVelocity stride:\t'+str(dn)+'\tsteps')
    log('\tMax frequency:\t\t'+str(np.round(thz[-1]/2,2))+'\tTHz')
    log('\tFrequency resolution:\t'+str(np.round((thz[1]-thz[0])*1e3,2))+'\tMHz')
    log('\tNo. of atoms:\t\t'+str(num)+'\t--')
    log('\tTotal No. of steps:\t'+str(steps)+'\t--')
    log('\tTotal time:\t\t'+str(np.round(steps*dt*1e9,2))+'\tns')
    log('\tTotal No. of splits:\t'+str(split)+'\t--')
    log('\tTime per split:\t\t'+str(np.round(steps/split*dt*1e12,2))+'\tps')
    log('\tNo. of K-points:\t'+str(nk)+'\t--')
    log('\tK-path:\t\t\t'+str(klabel))
    log('\t-----------------------------------')
    
##################################################################        
def getExp(nc,kvec,cellvec):
    """
    Creates array of exponential factor for every unit cell at each k point
    """
    exp = np.zeros(nc).astype(complex)
    for j in range(nc):
        exp[j] = np.exp(1j*kvec.dot(cellvec[j,2:5]))
        
    return exp
