import numpy as np

def write_output(phonons,params,lattice,eigen_vectors):
    if not params.with_eigs:
        np.savetxt(params.out_prefix+'.pSED',phonons.sed_avg,fmt='%.6f')
#        np.savetxt(params.out_prefix+'.PDoS',phonons.pdos,fmt='%.6f')
        np.savetxt(params.out_prefix+'.Qpts',lattice.reduced_qpoints,fmt='%.6f')
        np.savetxt(params.out_prefix+'.THz',phonons.thz,fmt='%.2f')
    else: 
        np.savetxt(params.out_prefix+'_BAND-TOTAL.pSED',phonons.sed_avg,fmt='%.6f')
#        np.savetxt(params.out_prefix+'.PDoS',phonons.pdos,fmt='%.6f')
        np.savetxt(params.out_prefix+'.Qpts',lattice.reduced_qpoints,fmt='%.6f')
        np.savetxt(params.out_prefix+'.THz',phonons.thz,fmt='%.2f')
        for i in range(len(phonons.sed_bands_avg[:,0,0])):
            np.savetxt(params.out_prefix+'_BAND-{}.pSED'
                    .format(i+1),phonons.sed_bands_avg[i,:,:],fmt='%.6f')
        np.savetxt(params.out_prefix+'_BAND-PHONOPY.disp',eigen_vectors.freq,
                fmt='%.6f')

class read_previous:
    def __init__(self,params):
        if params.with_eigs:
            if params.band_to_plot == 0:
                suffix = '_BAND-TOTAL'
            else:
                suffix = '_BAND-{}'.format(params.band_to_plot)
            self.sed_avg = np.loadtxt(params.out_prefix+suffix+'.pSED')
#            self.pdos =  np.loadtxt(params.out_prefix+'.PDoS')
            self.qpoints = np.loadtxt(params.out_prefix+'.Qpts')
            self.thz = np.loadtxt(params.out_prefix+'.THz')
            self.phonopy = np.loadtxt(params.out_prefix+'_BAND-PHONOPY.disp')

        else:
            self.sed_avg = np.loadtxt(params.out_prefix+'.pSED')
#            self.pdos =  np.loadtxt(params.out_prefix+'.PDoS')
            self.qpoints = np.loadtxt(params.out_prefix+'.Qpts')
            self.thz = np.loadtxt(params.out_prefix+'.THz')

def write_lorentz(lorentz,params):
    np.savetxt(params.out_prefix+'_LORENTZ-{}.params'.format(params.q_slice_index),
            lorentz.popt,fmt='%12f')
    np.savetxt(params.out_prefix+'_LORENTZ-{}.error'.format(params.q_slice_index),
            lorentz.pcov,fmt='%12f')
