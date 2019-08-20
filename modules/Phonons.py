import numpy as np

class spectral_energy_density:
    def __init__(self,params):
        self.construct_splits(params)
        self.sed = np.zeros((params.num_splits,
                            self.steps_per_split,
                            sum(params.num_qpoints)))
        self.thz = (np.arange(self.steps_per_split)/
                (self.steps_per_split*params.time_step*params.stride)/1e12)

    def construct_splits(self,params):
        self.reduced_steps = params.num_steps//params.stride
        self.steps_per_split = self.reduced_steps//params.num_splits

    def compute_sed(self,params,lattice):

        #### DEV ####
        print('\nDEV NOTE: the FFT\'s aren\'t properly scaled yet!\n')
        #### DEV ####

        self.num_unit_cells = lattice.unit_cells.max()
        self.num_basis = lattice.basis_pos.max()
        if params.debug:
            self.num_loops = 1
        else:
            self.num_loops = params.num_splits   
        self.loop_over_splits(params,lattice) # loop over each split, see next function
        self.sed_avg = self.sed.sum(axis=0)/self.num_loops
        
        # keep only the lower half
        max_freq = len(self.thz)//2
        self.sed_avg = self.sed_avg[:max_freq,:]
        self.thz = self.thz[:max_freq]

    def loop_over_splits(self,params,lattice):
        self.qdot = np.zeros((self.steps_per_split,sum(lattice.num_qpoints)))
        for i in range(self.num_loops):
            print('\nNow on split {} out {}...\n'.format(i+1,self.num_loops))
            self.loop_index = i
            self.get_simulation_data(params,lattice) # see end of file

            #### DEV ####
            # vdos ? # see end of file
            #### DEV ####

            self.loop_over_qpoints(params,lattice) # compute space FT, see next function

            #### DEV ####
            self.sed[i,:,:] = self.qdot/(4*np.pi) # scale this !!!
            #### DEV ####

    def loop_over_qpoints(self,params,lattice):
        for q in range(sum(lattice.num_qpoints)):
            self.q_index = q
            print('\tNow on q-point {} out of {}:\tq=({:.3f}, {:.3f}, {:.3f})'
                    .format(q+1,sum(lattice.num_qpoints),lattice.qpoints[q,0],
                        lattice.qpoints[q,1],lattice.qpoints[q,2]))
            self.exp_fac = np.tile(lattice.qpoints[q,:],(self.num_unit_cells,1))
            self.exp_fac = np.exp(1j*np.multiply(self.exp_fac,self.cell_vecs).sum(axis=1))
            self.loop_over_basis(params,lattice)

    def loop_over_basis(self,params,lattice):
        for i in range(self.num_basis):
            basis_ids = np.argwhere(lattice.basis_pos == (i+1)).reshape(
                    self.num_unit_cells)
            mass_arr = lattice.masses[basis_ids] 
            
            #### DEV ####
            vx = np.fft.fft(self.vels[:,basis_ids,0]
                    .reshape(self.steps_per_split,self.num_unit_cells)
                    *self.exp_fac*mass_arr,axis=0) # scale this !!!
            vy = np.fft.fft(self.vels[:,basis_ids,1]
                    .reshape(self.steps_per_split,self.num_unit_cells)
                    *self.exp_fac*mass_arr,axis=0) # scale this !!!
            vz = np.fft.fft(self.vels[:,basis_ids,2]
                    .reshape(self.steps_per_split,self.num_unit_cells)
                    *self.exp_fac*mass_arr,axis=0) # scale this !!!
            #### DEV ####

            self.qdot[:,self.q_index] = (self.qdot[:,self.q_index]+
                    (abs(vx.sum(axis=1))**2+abs(vy.sum(axis=1))**2+
                        abs(vz.sum(axis=1))**2)/self.num_unit_cells)   
            
    def get_simulation_data(self,params,lattice):
        self.vels = params.database['vels'][self.loop_index*self.steps_per_split:
                (self.loop_index+1)*self.steps_per_split,:,:]
        self.pos = params.database['pos'][self.loop_index*self.steps_per_split:
                (self.loop_index+1)*self.steps_per_split,:,:]
        
        #### DEV ####
        # time average the positions (for now, maybe can do corr. between pos and vels)
        self.cell_vecs = self.pos[:,lattice.cell_ref_ids,:].mean(axis=0) 
        #### DEV ####


