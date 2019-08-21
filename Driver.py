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

#########################################################

input_file = 'INPUT'
params = Parsers.parse_input(input_file)

if params.plot_bands:
    sed_avg, qpoints, thz = FileIO.read_previous(params)
    Plot.plot_bands(sed_avg,qpoints,thz)
    if not params.plot_slice:
        exit()
if params.plot_slice:
    sed_avg, qpoints, thz = FileIO.read_previous(params)
    Plot.plot_slice(sed_avg,qpoints,thz,params.q_slice)
    exit()
if params.compress:
    Compressor.compress(params)
    exit()

params.database = h5py.File(params.database_file,'r')
lattice = Lattice.lattice(params)

phonons = Phonons.spectral_energy_density(params)
phonons.compute_sed(params,lattice)

FileIO.write_output(phonons,params,lattice)
