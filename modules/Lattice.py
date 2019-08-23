import numpy as np
import Parsers

class lattice:
    def __init__(self,params,eigen_vectors):

        # read the unitcell and basis positions from a file
        self.atom_ids, self.unit_cells, self.basis_pos, self.masses = (Parsers.
                parse_lattice_file(params))
        self.cell_ref_ids = np.argwhere(self.basis_pos == 1)
        self.cell_ref_ids = self.cell_ref_ids.reshape(len(self.cell_ref_ids))

        # calculate direct lattice vectors
        dir_lat = params.prim_vecs
        dir_lat[:,0] = dir_lat[:,0]*params.lat_params[0]
        dir_lat[:,1] = dir_lat[:,1]*params.lat_params[1]
        dir_lat[:,2] = dir_lat[:,2]*params.lat_params[2]

        # calculate reciprocal lattice vectors
        self.bz_vol = dir_lat[0,:].dot(np.cross(dir_lat[1,:],dir_lat[2,:]))
        self.recip_vecs = np.zeros((3,3))
        self.recip_vecs[0,:] = 2*np.pi*np.cross(dir_lat[1,:],dir_lat[2,:])/self.bz_vol
        self.recip_vecs[1,:] = 2*np.pi*np.cross(dir_lat[2,:],dir_lat[0,:])/self.bz_vol
        self.recip_vecs[2,:] = 2*np.pi*np.cross(dir_lat[0,:],dir_lat[1,:])/self.bz_vol
        
        if params.with_eigs: # use phonopy q-points
            print('\nCOMMENT: Getting q-points from PHONOPY \'{}\' file\n'
                    .format(params.eigvecs_file))
            params.num_qpoints = list(map(int,[eigen_vectors.num_qpoints]))
            self.num_qpoints = list(map(int,[eigen_vectors.num_qpoints]))
            # phonopy provides reduced q-points
            self.reduced_qpoints = eigen_vectors.qpoints
            self.qpoints = np.copy(self.reduced_qpoints)
            # convert to recip. lattice-vector units
            for i in range(sum(self.num_qpoints)):
                self.qpoints[i,:] = self.recip_vecs.dot(self.qpoints[i,:])

        else: # create q-point list from input file
            # construct the BZ paths
            self.construct_BZ_path(params)

        # print list of q-points to screen
        print('\nThere are {} q-points to be calculated:\n'
                    .format(sum(params.num_qpoints)))
        for i in range(sum(params.num_qpoints)):
            print('\t(reduced) q=({:.4f}, {:.4f}, {:.4f})'
                    .format(self.reduced_qpoints[i,0],self.reduced_qpoints[i,1],
                        self.reduced_qpoints[i,2]))



    def construct_BZ_path(self,params):
        # convert reduced q-points to recip. lattice-vector units
        self.qsym_points = np.copy(params.qsym_points)
        for i in range(len(params.qsym_points[:,0])):
            self.qsym_points[i,:] = self.recip_vecs.dot(params.qsym_points[i,:])
        self.num_qpoints = params.num_qpoints
        self.qpoints = np.zeros((sum(self.num_qpoints),3)) 
        self.reduced_qpoints = np.zeros((sum(self.num_qpoints),3))

        # make q-points list by linear interpolation between symmetry points
        start = 0
        for i in range(len(self.num_qpoints)):
            end = start+self.num_qpoints[i]    
            for j in range(3):
                self.qpoints[start:end,j] = np.linspace(self.qsym_points[i,j],
                        self.qsym_points[(i+1),j],self.num_qpoints[i],endpoint=False)
                self.reduced_qpoints[start:end,j] = np.linspace(params.qsym_points[i,j],
                        params.qsym_points[(i+1),j],self.num_qpoints[i],endpoint=False)
            start = start+self.num_qpoints[i]
                 
