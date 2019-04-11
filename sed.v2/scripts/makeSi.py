#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:33:02 2019

@author: ty
"""

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
    import copy as cp
    
    nx = 50 
    ny = 2
    nz = 2
    lammps='data.relax'
    
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
                fid.write('\n' + str(i+1) + ' ' + str(mass))
                fid.write('\n\nAtoms\n\n')
                for i in range(num-1):
                    fid.write(str(int(i+1)) + ' ' + str(int(pos[i,1])) + ' ' 
                              + str(pos[i,2]) + ' ' +
                            str(pos[i,3]) + ' ' + str(pos[i,4]) + '\n')
                fid.write(str(len(pos)) +  ' ' + str(int(pos[-1,1])) + ' ' 
                          + str(pos[-1,2]) + ' ' +
                        str(pos[-1,3]) + ' ' + str(pos[-1,4]))
    return [num, pos, mass, uc, ids, a]