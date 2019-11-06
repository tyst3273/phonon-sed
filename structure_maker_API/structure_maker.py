import sys
import numpy as np

modulepath = modulepath = '../modules' # CHANGE THIS TO YOUR PATH
sys.path.append(modulepath)

import Lattice


si = Lattice.structure_maker('10x2x2 Silicon')

si.basis([[0.00, 0.00, 0.00],
          [0.00, 0.50, 0.50],
          [0.50, 0.00, 0.50],
          [0.50, 0.50, 0.00],
          [0.25, 0.25, 0.25],
          [0.75, 0.75, 0.25],
          [0.25, 0.75, 0.75],
          [0.75, 0.25, 0.75]], # basis coords
          ['Si','Si','Si','Si','Si','Si','Si','Si'], # types of atoms
          masses=[28.0855], # AMU, 1 per type
          reduced_coords=True) # in crytal coords, False for Cartesian

si.lattice_vectors([[ 1, 0, 0], 
                    [ 0, 1, 0],
                    [ 0, 0, 1]],
                    lattice_constants=[5.431,5.431,5.431]) # angstrom

si.replicate([16,2,2]) # size of supercell, here it is 16x2x2

# si.write_xyz('si.xyz') # xyz file, can use it to plot in VMD
si.write_lammps('si.lammps') # LAMMPS input file
si.write_lattice_file('lattice.dat') # file needed for the pSED code

