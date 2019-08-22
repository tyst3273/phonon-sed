import numpy as np
from scipy.optimize import leastsq

class lorentz:
    def __init__(self,data,params):
        self.find_nearest(data,params)
        self.sed = data.sed_avg[:,self.q_ind]
#        self.find_peaks(data)
    
#    def find_peaks(self,data,params):
        


    def find_nearest(self,data,params):
        nearest = ['','','']
        nearest[0] = min(data.qpoints[:,0], key=lambda x:abs(x-params.q_slice[0]))
        inds = np.argwhere(data.qpoints[:,0] == nearest[0]).flatten()
        if len(inds) == 1:
            q_ind = inds[0]
        else:
            qpt_slice = data.qpoints[inds,:]
            nearest[1] = min(qpt_slice[:,1], key=lambda x:abs(x-params.q_slice[1]))
            inds2 = np.argwhere(data.qpoints[:,1] == nearest[1]).flatten()
            inds = np.intersect1d(inds,inds2)
            if len(inds) == 1:
                q_ind = inds[0]
            else:
                qpt_slice = data.qpoints[inds,:]
                nearest[2] = min(qpt_slice[:,2], key=lambda x:abs(x-params.q_slice[2]))
                inds2 = np.argwhere(data.qpoints[:,2] == nearest[2]).flatten()
                inds = np.intersect1d(inds,inds2)
                q_ind = inds[0]
        self.nearest = data.qpoints[q_ind,:]
        self.q_ind = q_ind

        
        
