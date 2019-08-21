import numpy as np
import os
import h5py

def print_error(txt):
    print('\nERROR: value for input paramater {} seems wrong\n'.format(txt))
    exit()

class parse_input:
    def __init__(self,input_file):
        
        ### defaults
        self.compress = False
        self.debug = False
        self.plot_previous = False
        self.plot_slice = False
        self.slice = [0.0, 0.0, 0.0]
        self.num_bins = 1
        self.out_prefix = 'last-run'
        self.vels_file = 'vels.dat'
        self.pos_file = 'pos.dat'
        self.lattice_file = 'lattice.dat'
        self.database_file = 'dat.hdf5'
        self.file_format = 'xyz'

        self.input_file = input_file
        input_txt = open(input_file,'r').readlines()
        for line in input_txt:
            txt = line.strip()

            # skip blank and comment lines
            if len(line.strip()) == 0: 
                continue
            elif line.strip()[0] == '#':
                continue
            
            # crystal info
            txt = txt.strip().split()
            if txt[0] == 'NUM_ATOMS':
                try:
                    self.num_atoms = int(txt[txt.index('=')+1])
                except:
                    print_error('NUM_ATOMS')
            elif txt[0] == 'LAT_PARAMS':
                try:
                    self.lat_params = np.array(txt[(txt.index('=')+1):
                                            (txt.index('=')+4)]).astype(float)
                except:
                    print_error('LAT_PARAMS')
            elif txt[0] == 'PRIM_VECS':
                try:
                    self.prim_vecs = np.array(txt[(txt.index('=')+1):
                        (txt.index('=')+10)]).astype(float)
                    self.prim_vecs = self.prim_vecs.reshape(3,3)
                except:
                    print_error('PRIM_VECS')
            
            # Q-points
            elif txt[0] == 'NUM_QPATHS':
                try:
                    self.num_qpaths = int(txt[txt.index('=')+1])
                except:
                    print_error('NUM_QPATHS')
            elif txt[0] == 'NUM_QPOINTS':
                try:
                    self.num_qpoints = np.array(txt[(txt.index('=')+1):
                        (txt.index('=')+self.num_qpaths+1)]).astype(int)
                except:
                    print_error('NUM_QPOINTS')
            elif txt[0] == 'QSYM_POINTS':
                try:
                    self.qsym_points = np.array(txt[(txt.index('=')+1):
                        (txt.index('=')+int((self.num_qpaths+1)*3+1))]).astype(float)
                    self.qsym_points = self.qsym_points.reshape(self.num_qpaths+1,3)
                except:
                    print_error('QSYM_POINTS')

            # simulation control parameters
            elif txt[0] == 'NUM_STEPS':
                try:
                    self.num_steps = int(txt[txt.index('=')+1])
                except:
                    print_error('NUM_STEPS')
            elif txt[0] == 'STRIDE':
                try:
                    self.stride = int(txt[txt.index('=')+1])
                except:
                    print_error('STRIDE')
            elif txt[0] == 'NUM_SPLITS':
                try:
                    self.num_splits = int(txt[txt.index('=')+1])
                except:
                    print_error('NUM_SPLITS')
            elif txt[0] == 'TIME_STEP':
                try:
                    self.time_step = float(txt[txt.index('=')+1])
                except:
                    print_error('TIME_STEP')
                self.time_step = self.time_step*1e-15

            # whether or not to build hdf5 database
            elif txt[0] == 'COMPRESS':
                try:
                    self.compress = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('COMPRESS')

            # debug info
            elif txt[0] == 'DEBUG':
                try:
                    self.debug = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('DEBUG')

            # plotting
            elif txt[0] == 'PLOT_BANDS':
                try:
                    self.plot_bands = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('PLOT_BANDS')
            elif txt[0] == 'PLOT_SLICE':
                try:
                    self.plot_slice = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('PLOT_SLICE')
            elif txt[0] == 'Q_SLICE':
                try:
                    self.q_slice = list(map(float,txt[txt.index('=')+1:txt.index('=')+4]))
                except:
                    print_error('Q_SLICE')
            elif txt[0] == 'NUM_BINS':
                try:
                    self.num_bins = int(txt[txt.index('=')+1]) 
                except:
                    print_error('NUM_BINS')


            # file names
            elif txt[0] == 'FILE_FORMAT':
                try:
                    self.file_format = str(txt[txt.index('=')+1].strip('\''))
                except:
                    print_error('FILE_FORMAT')
            elif txt[0] == 'VELS_FILE':
                try:
                    self.vels_file = str(txt[txt.index('=')+1].strip('\''))
                except:
                    print_error('VELS_FILE')
            elif txt[0] == 'POS_FILE':
                try:
                    self.pos_file = str(txt[txt.index('=')+1].strip('\''))
                except:
                    print_error('POS_FILE')
            elif txt[0] == 'LATTICE_FILE':
                try:
                    self.lattice_file = str(txt[txt.index('=')+1].strip('\''))
                except:
                    print_error('LATTICE_FILE')
                if not os.path.exists(self.lattice_file):
                    print('\nERROR: file {} not found\n'.format(self.lattice_file))
                    exit()
            elif txt[0] == 'OUT_PREFIX':
                try:
                    self.out_prefix = str(txt[txt.index('=')+1].strip('\''))
                except:
                    print_error('OUT_PREFIX')
            elif txt[0] == 'EIGVECS_FILE':
                try:
                    self.eigvecs_file = str(txt[txt.index('=')+1].strip('\''))
                except:
                    print_error('EIGVECS_FILE')
                if not os.path.exists(self.eigvecs_file):
                    print('\nERROR: file {} not found\n'.format(self.eigvecs_file))
#                    exit()

            # unknown options
#            else: 
#                print('\nERROR: option {} not recognized\n'.format(txt[0]))
#                exit()

def parse_lattice_file(params):

        # read the lattice info from a file
        atom_ids, unit_cells, basis_pos, masses = np.loadtxt(
                params.lattice_file,unpack=True)

        # set proper data types
        atom_ids = atom_ids.astype(int)
        unit_cells = unit_cells.astype(int)
        basis_pos = basis_pos.astype(int)
        masses = masses.astype(float)

        return atom_ids, unit_cells, basis_pos, masses



