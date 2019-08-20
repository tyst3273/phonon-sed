import numpy as np
import matplotlib.pyplot as plt

def plot_sed(sed_avg,qpoints,thz):

    log = True
    color = 'inferno'
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

    #plt.savefig('example.png',format='png',dpi=300,bbox_inches='tight')

    plt.show()

