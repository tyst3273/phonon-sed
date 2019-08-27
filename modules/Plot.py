import numpy as np
import matplotlib.pyplot as plt

def plot_bands(data,params):
    sed_avg = data.sed_avg
    qpoints = data.qpoints
    thz = data.thz

    log = True
    color = 'afmhot'  #'inferno'
    interp = 'hamming'
    df = 5

    if log:
        sed_avg = np.log(sed_avg)

    ### creat a figure, set its size
    fig, ax = plt.subplots()
    fig.set_size_inches(4,6,forward=True)
    fig.tight_layout(pad=5)

    ax.imshow(sed_avg,cmap=color,interpolation=interp,aspect='auto',origin='lower')

    # configure the plot
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    freqs = np.arange(0,thz.max(),df)
    nf = len(freqs)
    ids = np.zeros(nf)
    for i in range(nf):
        ids[i] = np.argwhere(thz <= freqs[i]).max()
    ax.set_yticks(ids)
    ax.set_yticklabels(list(map(str,freqs)))

    xticks = [0,len(qpoints)-1]
    ax.set_xticks(xticks)
    xlabels = ['']*len(xticks)
    for i in range(len(xticks)):
        xlabels[i] = '({:.1f},{:.1f},{:.1f})'.format(qpoints[xticks[i],0],
                qpoints[xticks[i],1],qpoints[xticks[i],2])
    ax.set_xticklabels(xlabels)  

    ax.minorticks_on()
    ax.tick_params(which='both', axis='y', width=1, labelsize='large')
    ax.tick_params(which='both', axis='x', width=1, labelsize='small',
            labelrotation=0.0,pad=5.0)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3, color='k')
#    plt.tick_params(axis='x',which='both',labelbottom=False)
    ax.set_xlabel(r'$\bfq$',labelpad=5.0,fontweight='normal',fontsize='x-large')
    ax.set_ylabel(r'$\omega$ (THz)',labelpad=3.0,fontweight='normal',fontsize='x-large')
    fig.suptitle(r'$\Phi$($\bfq$,$\omega)$',y=0.95,fontsize='x-large')

    plt.savefig('example.png',format='png',dpi=300,bbox_inches='tight')
    plt.show()



def lorentzian(xarr,center,amplitude,hwhm):
            return amplitude/(1+((xarr-center)/hwhm)**2)

def plot_slice(data,params):
    sed_avg = data.sed_avg
    qpoints = data.qpoints
    thz = data.thz
    q_ind = params.q_slice_index

    log = True

    df = 2
    freqs = np.arange(0,thz.max(),df)
    nf = len(freqs)
    ids = np.zeros(nf)
    for i in range(nf):
        ids[i] = np.argwhere(thz <= freqs[i]).max()

    ### creat a figure, set its size
    fig, ax = plt.subplots()
    fig.set_size_inches(8,4,forward=True)
    fig.tight_layout(pad=8)
    
    # configure the plot
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    if log:
        ax.semilogy(sed_avg[:,q_ind],ls='-',lw=1,color='k',
                marker='o',ms=2,mfc='b',mec='k',mew=1)
        if params.plot_lorentz:
            total = np.zeros(len(sed_avg[:,q_ind]))
            for i in range(len(params.popt[:,0])):
                start = params.bounds[i,0]
                end = params.bounds[i,1]
                if params.popt[i,2] == 0:
                    continue
                ax.semilogy(lorentzian(np.arange(len(sed_avg[:,q_ind])),
                    params.popt[i,0],params.popt[i,1],params.popt[i,2]),
                    ls='-',lw=1,marker='o',mfc='r',mec='r',ms=1,mew=0,color='r')
                total = total+(lorentzian(np.arange(len(sed_avg[:,q_ind])),
                    params.popt[i,0],params.popt[i,1],params.popt[i,2]))
            ax.semilogy(total,ls='-',lw=0.5,marker='o',
                    mfc='b',mec='b',ms=0.5,mew=0,color='b')

    else:
        ax.plot(sed_avg[:,q_ind],ls='-',lw=1,color='k',
                marker='o',ms=2,mfc='b',mec='k',mew=1)
        if params.plot_lorentz:
            total = np.zeros(len(sed_avg[:,q_ind]))
            for i in range(len(params.popt[:,0])): 
                start = params.bounds[i,0]
                end = params.bounds[i,1]
                if params.popt[i,2] == 0:
                    continue
                ax.plot(lorentzian(np.arange(len(sed_avg[:,q_ind])),
                    params.popt[i,0],params.popt[i,1],params.popt[i,2]),
                    ls='-',lw=1,marker='o',mfc='r',mec='r',ms=1,mew=0,color='r')
                total = total+(lorentzian(np.arange(len(sed_avg[:,q_ind])),
                    params.popt[i,0],params.popt[i,1],params.popt[i,2]))
            ax.plot(total,ls='-',lw=0.5,marker='o',mfc='b',mec='b',
                    ms=0.5,mew=0,color='b')

    ax.minorticks_on()
    ax.tick_params(which='both', width=1, labelsize='x-large')
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3, color='k')
#    plt.tick_params(axis='x',which='both',labelbottom=False)
    ax.set_ylabel(r'log($\Phi$($\omega)$)',labelpad=35.0,fontweight='normal',
            fontsize='x-large',rotation='horizontal')
    ax.set_xlabel('Index',labelpad=3.0,fontweight='normal',fontsize='large')
#    ax.set_xlabel(r'$\omega$ (THz)',labelpad=3.0,fontweight='normal',fontsize='large')
    fig.suptitle(r'$\bfq$=({:.3f}, {:.3f}, {:.3f})'.format(
        qpoints[q_ind,0],qpoints[q_ind,1],qpoints[q_ind,2]),y=0.80,fontsize='x-large')

    #plt.savefig('example.png',format='png',dpi=300,bbox_inches='tight')
    plt.show()

    
