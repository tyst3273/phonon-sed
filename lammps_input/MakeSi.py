import numpy as np
import sys
import copy as cp

nx = 24
ny = 4
nz = 4

a = 5.431
mass = 28.0855
    
basis = np.array([[0,0,0], 
                  [0,2,2],
                  [2,0,2],
                  [2,2,0],
                  [1,1,1],
                  [3,3,1],
                  [1,3,3],
                  [3,1,3]])
    
types = np.ones(8).astype(int)
uc = np.ones(8).astype(int)
ids = np.arange(1,8+1).astype(int)

pos = np.copy(basis)
tmp = np.copy(pos)
tmp_uc = np.copy(uc)
tmp_ids = np.copy(ids)
for i in range(nx-1): #x
    tmp[:,0] = tmp[:,0]+4
    pos = np.append(pos,tmp,axis=0)
    tmp_uc = tmp_uc+1
    uc = np.append(uc,tmp_uc)
    ids = np.append(ids,tmp_ids)

tmp = np.copy(pos)
tmp_uc = np.copy(uc)
tmp_ids = np.copy(ids)
for i in range(ny-1): 
    tmp[:,1] = tmp[:,1]+4
    pos = np.append(pos,tmp,axis=0)
    tmp_uc = tmp_uc+nx
    uc = np.append(uc,tmp_uc)
    ids = np.append(ids,tmp_ids)

tmp = np.copy(pos)
tmp_uc = np.copy(uc)
tmp_ids = np.copy(ids)
for i in range(nz-1): 
    tmp[:,2] = tmp[:,2]+4
    pos = np.append(pos,tmp,axis=0)
    tmp_uc = tmp_uc+nx*ny
    uc = np.append(uc,tmp_uc)
    ids = np.append(ids,tmp_ids)

num_atoms = len(pos[:,0])
atoms_ids = np.arange(1,num_atoms+1).reshape(num_atoms,1)
pos = np.append(np.ones(num_atoms).reshape(num_atoms,1),pos,axis=1)
pos = np.append(atoms_ids,pos,axis=1)
pos[:,2] = pos[:,2]*a/4
pos[:,3] = pos[:,3]*a/4
pos[:,4] = pos[:,4]*a/4  

pos[:,2:] = np.round(pos[:,2:],6)

with open('data.pos', 'w') as fid:
        buff = a/8

        xmax = pos[:,2].max()+buff
        xmin = 0-buff
        ymax = pos[:,3].max()+buff
        ymin = 0-buff
        zmax = pos[:,4].max()+buff
        zmin = 0-buff 
        
        fid.write(str('LAMMPS pos file\n'))
        fid.write('\n' + str(num_atoms) + ' atoms\n')
        fid.write('\n1 atom types\n')
        fid.write('\n' + str(xmin)+' '+str(xmax)+' xlo'+' xhi\n')
        fid.write(str(ymin)+' '+str(ymax)+' ylo'+' yhi\n')
        fid.write(str(zmin)+' '+str(zmax)+' zlo'+' zhi\n')
        fid.write('\nMasses\n')
        fid.write('\n1 ' + str(mass))
        fid.write('\n\nAtoms\n\n')
        for i in range(num_atoms-1):
            fid.write(str(int(i+1)) + ' ' + str(int(pos[i,1])) + ' ' 
                              + str(pos[i,2]) + ' ' +
                            str(pos[i,3]) + ' ' + str(pos[i,4]) + '\n')
        fid.write(str(len(pos)) +  ' ' + str(int(pos[-1,1])) + ' ' 
                          + str(pos[-1,2]) + ' ' +
                        str(pos[-1,3]) + ' ' + str(pos[-1,4]))

uc = uc.reshape(num_atoms,1)
ids = ids.reshape(num_atoms,1)

savedat = np.append(np.append(np.append(atoms_ids,uc,axis=1),ids,axis=1),
        np.ones(num_atoms).reshape(num_atoms,1)*mass,axis=1)

np.savetxt('lattice.dat',savedat,fmt='%d %d %d %.4f')
