import numpy as np
import Parsers

class lattice:
    def __init__(self,params):

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

        # construct the BZ paths
        self.construct_BZ_path(params)

    def construct_BZ_path(self,params):
        # convert from reduced q-points
        self.qsym_points = np.copy(params.qsym_points)
        for i in range(len(params.qsym_points[:,0])):
            self.qsym_points[i,:] = self.recip_vecs.dot(params.qsym_points[i,:])
    
        # populate the BZ path-mesh with linear interpolation between points
        self.num_qpoints = params.num_qpoints
        self.qpoints = np.zeros((sum(self.num_qpoints),3)) 
        start = 0
        for i in range(len(self.num_qpoints)):
            end = start+self.num_qpoints[i]

            for j in range(3):
                self.qpoints[start:end,j] = np.linspace(self.qsym_points[i,j],
                        self.qsym_points[(i+1),j],self.num_qpoints[i],endpoint=False)

            start = start+self.num_qpoints[i]
                 
