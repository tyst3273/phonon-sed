import numpy as np
import Plot
import FileIO
from scipy.optimize import curve_fit

class lorentz:
    def __init__(self,data,params):

        def lorentzian(xarr,center,amplitude,hwhm):
            return amplitude/(1+((xarr-center)/hwhm)**2)

        self.find_nearest(data,params)
        self.sed = data.sed_avg[:,self.q_ind]

        dx = 10
        dxarr = 50
        maxfev = 1e6

        self.xarr = np.arange(len(self.sed))
        self.popt = np.zeros((params.num_guesses,3))
        params.bounds = np.zeros((params.num_guesses,2))

        for i in range(params.num_guesses):
            if params.peak_guesses[i]-dxarr < 0:
                start = 0
            else:
                start = params.peak_guesses[i]-dxarr
            if params.peak_guesses[i]+dxarr > len(self.sed)-1:
                end = len(self.sed)-1
            else:
                end = params.peak_guesses[i]+dxarr
            params.bounds[i,0] = start
            params.bounds[i,1] = end
            lb = [params.peak_guesses[i]-dx,1e-6,1e-10]
            ub = [params.peak_guesses[i]+dx,np.inf,1e3]

            self.popt[i,:], pcov = curve_fit(lorentzian,
                    self.xarr[start:end],
                    self.sed[start:end],
                    p0=[params.peak_guesses[i],self.sed[params.peak_guesses[i]],1],
                    bounds=(lb,ub),maxfev=maxfev)
        
        FileIO.write_lorentz(self,params)

        params.popt = self.popt
        params.plot_lorentz = True
        Plot.plot_slice(data,params)


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

        
        
