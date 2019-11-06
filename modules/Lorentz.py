import numpy as np
import Plot
import FileIO
from scipy.optimize import curve_fit

class lorentz:
    def __init__(self,data,params):

        def lorentzian(xarr,center,amplitude,hwhm):
            return amplitude/(1+((xarr-center)/hwhm)**2)

        self.q_ind = params.q_slice_index
        self.sed = data.sed_avg[:,self.q_ind]

        # some bounds on the fitting. Might need to tweak these
        dx = 10
        dxarr = 100
        maxfev = 1e4

        self.xarr = np.arange(len(self.sed))
        self.popt = np.zeros((params.num_guesses,3))
        self.pcov = np.zeros((params.num_guesses,3))
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

            try:
                self.popt[i,:], pcov = curve_fit(lorentzian,
                        self.xarr[start:end],
                        self.sed[start:end],
                        p0=[params.peak_guesses[i],self.sed[params.peak_guesses[i]],1],
                        bounds=(lb,ub),maxfev=maxfev)
                self.pcov[i,:] = np.sqrt(np.diag(pcov))
            except:
                print('\nWARNING: Lorentz fit for peak-guess-index {} failed!\n'
                        .format(params.peak_guesses[i]))
                continue
        
        FileIO.write_lorentz(self,params)

        params.popt = self.popt
        params.plot_lorentz = True
        Plot.plot_slice(data,params)

