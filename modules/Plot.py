import numpy as np
import matplotlib.pyplot as plt

def plot_bands(sed_avg,qpoints,thz):

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

    ax.minorticks_on()
    ax.tick_params(which='both', width=1, labelsize='x-large')
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3, color='k')
    plt.tick_params(axis='x',which='both',labelbottom=False)

    ax.set_xlabel(r'$\bfq$',labelpad=8.0,fontweight='normal',fontsize='x-large')
    ax.set_ylabel(r'$\omega$ (THz)',labelpad=3.0,fontweight='normal',fontsize='x-large')

    fig.suptitle(r'$\Phi$($\bfq$,$\omega)$',y=0.95,fontsize='x-large')

    plt.savefig('example.png',format='png',dpi=300,bbox_inches='tight')

    plt.show()



def plot_slice(sed_avg,qpoints,thz,q_slice):

    log = True
    df = 5

    freqs = np.arange(0,thz.max(),df)
    nf = len(freqs)
    ids = np.zeros(nf)

    nearest = ['','','']
    nearest[0] = min(qpoints[:,0], key=lambda x:abs(x-q_slice[0]))
    inds = np.argwhere(qpoints[:,0] == nearest[0]).flatten()
    if len(inds) == 1:
        q_ind = inds[0]
    else:
        qpt_slice = qpoints[inds,:]
        nearest[1] = min(qpt_slice[:,1], key=lambda x:abs(x-q_slice[1]))
        inds2 = np.argwhere(qpoints[:,1] == nearest[1]).flatten()
        inds = np.intersect1d(inds,inds2)
        if len(inds) == 1:
            q_ind = inds[0]
        else:
            qpt_slice = qpoints[inds,:]
            nearest[2] = min(qpt_slice[:,2], key=lambda x:abs(x-q_slice[2]))
            inds2 = np.argwhere(qpoints[:,2] == nearest[2]).flatten()
            inds = np.intersect1d(inds,inds2)
            q_ind = inds[0]
    nearest = qpoints[q_ind,:]

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
    else:
        ax.plot(sed_avg[:,q_ind],ls='-',lw=1,color='k',
                marker='o',ms=2,mfc='b',mec='k',mew=1)
    
    for i in range(nf):
        ids[i] = np.argwhere(thz <= freqs[i]).max()

    ax.set_xticks(ids)
    ax.set_xticklabels(list(map(str,freqs)))

    ax.minorticks_on()
    ax.tick_params(which='both', width=1, labelsize='x-large')
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3, color='k')

#    plt.tick_params(axis='x',which='both',labelbottom=False)

    ax.set_ylabel(r'$\Phi$($\omega)$',labelpad=30.0,fontweight='normal',
            fontsize='x-large',rotation='horizontal')
    ax.set_xlabel(r'$\omega$ (THz)',labelpad=3.0,fontweight='normal',fontsize='x-large')

    fig.suptitle(r'$\bfq$=({}, {}, {}), log-scale={}'
            .format(nearest[0],nearest[1],nearest[2],log),y=0.80,fontsize='x-large')

    #plt.savefig('example.png',format='png',dpi=300,bbox_inches='tight')

    plt.show()

    
