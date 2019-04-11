#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is Ty's custom module to for the SED code: keep it in the same
directory

DATE STAMP: 02.20.2019 MM.DD.YYYY

now includes 'makeGaN'
"""
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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
    
    return [kpoints, kdist]

##########################################################
def makeTriclinic(n1,n2,n3,lammps='no',element='si'):
    """
    See docstring for makeFCCdiamond
    """
    import numpy as np
    import sys
    import copy as cp
    
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
    import numpy as np
    import sys
    import copy as cp

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
    import numpy as np
    import copy as cp
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
    import numpy as np
    import copy as cp

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

##########################################################
def writeSED(outfile,thz,kpoints,sed,dos):
    """
    This function is simple. It writes the frequency data array, k points, 
    SED matrix, and DOS array to file 'outfile'. Read file with readSED
    """
    import numpy as np
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
        import numpy as np
        nf = int(fid.readline().strip().split()[2])
        nk = int(fid.readline().strip().split()[2])
        thz = np.zeros((nf,1))
        kpoints = np.zeros((nk,3))
        sed = np.zeros((nf*nk,1)).astype(complex)
        dos = np.zeros((nf,1))
        for i in range(nf):
            thz[i,0] = float(fid.readline())
        for i in range(nk):
            kpoints[i,:] = fid.readline().strip().split()[:]
        kpoints = kpoints.astype(float)
        for i in range(nf*nk):
            sed[i,0] = complex(fid.readline())
        sed = np.reshape(sed,(nf,nk))
        for i in range(nf):
            dos[i,0] = float(fid.readline())
    return thz, kpoints, sed, dos

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
    import numpy as np
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
def vdos(vels,tn,num,dt,dn,win,thz):
    """
    This function calculates vibrational density of states from EMD velocity
    data. Intended to be used adjunct to SED code. 
    """
    import numpy as np
    vels = vels.reshape(tn,num*3)
    dos = np.zeros((tn,num*3))
    velsfft = np.fft.fft(vels,axis=0)*dt*dn
    dos = (np.multiply(abs(velsfft),abs(velsfft))/
               np.tile(np.multiply(vels,vels).mean(axis=0),(tn,1))/(tn*dt*dn))
    dosx = gsmooth(dos[:,0::3].mean(axis=1),win,(thz[1]-thz[0])*2*np.pi*1e12)
    dosy = gsmooth(dos[:,1::3].mean(axis=1),win,(thz[1]-thz[0])*2*np.pi*1e12)
    dosz = gsmooth(dos[:,2::3].mean(axis=1),win,(thz[1]-thz[0])*2*np.pi*1e12)
    dos = gsmooth(dos.mean(axis=1),win,(thz[1]-thz[0])*2*np.pi*1e12)

    return [dos, dosx, dosy, dosz]

##########################################################
def log(string,outfile='log.txt',suppress='no',new='no'):
    """
    This function prints output to a file called log.txt and to the screen. 
    Useful for tracking whats happening when submitted using qsub or slurm etc.
    If you don't want to print to screen, enter supress='yes'
    """
    if new == 'no':
        with open(outfile,'a') as fid:
            fid.write(string)
    else:
        with open(outfile,'w') as fid:
            fid.write(string)
    if suppress == 'no':
        print(string)
    
        
