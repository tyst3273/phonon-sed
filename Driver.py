# python modules
import sys
import numpy as np
import h5py

# add my modules to path
modulepath = '/home/ty/python_modules/sed_modules'
sys.path.append(modulepath)

# my modules
import Parsers
import Compressor
import Lattice
import Phonons
import FileIO
import Plot
import Lorentz

#########################################################

input_file = 'INPUT'
params = Parsers.parse_input(input_file)

if params.plot_bands:
    data = FileIO.read_previous(params)
    Plot.plot_bands(data,params)
    if not params.plot_slice and not params.lorentz:
        print('\nALL DONE!\n')
        exit()
if params.plot_slice:
    data = FileIO.read_previous(params)
    Plot.plot_slice(data,params)
    exit()
if params.lorentz:
    data = FileIO.read_previous(params)
    Lorentz.lorentz(data,params)
    print('\nALL DONE!\n')
    exit()

if params.compress:
    Compressor.compress(params)
    print('\nALL DONE!\n')
    exit()

params.database = h5py.File(params.database_file,'r')

eigen_vectors = Parsers.parse_eigen_vecs(params)
lattice = Lattice.lattice(params,eigen_vectors)

phonons = Phonons.spectral_energy_density(params)
phonons.compute_sed(params,lattice,eigen_vectors)

FileIO.write_output(phonons,params,lattice,eigen_vectors)

print('\nALL DONE!\n')
