import numpy as np

def write_output(phonons,params,lattice):
    np.savetxt(params.out_prefix+'.pSED',phonons.sed_avg,fmt='%.6f')
#    np.savetxt(params.out_prefix+'.PDoS',phonons.pdos,fmt='%.6f')
    np.savetxt(params.out_prefix+'.Qpts',lattice.qpoints,fmt='%.4f')
    np.savetxt(params.out_prefix+'.THz',phonons.thz,fmt='%.2f')

def read_previous(params):
    sed_avg = np.loadtxt(params.out_prefix+'.pSED')
#    pdos =  np.loadtxt(params.out_prefix+'.PDoS')
    qpoints = np.loadtxt(params.out_prefix+'.Qpts')
    thz = np.loadtxt(params.out_prefix+'.THz')

    return [sed_avg, qpoints, thz]

