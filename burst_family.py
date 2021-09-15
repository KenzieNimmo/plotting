from load_file import load_archive
from fluence import fluences

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
import os 

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'

def normalise(ds, t_cent, t_sig):
    """
    Calibrate the dynamic spectrum for the bandpass

    Per frequency channel it subtracts the off burst mean and divides by the off burst std
    """
    ds_off = np.concatenate((ds[:,0:int(t_cent-3*t_sig)],ds[:,int(t_cent+3*t_sig):]),axis=1)
    
    for chan in range(ds_off.shape[0]):
        ds[chan,:] -= np.mean(ds_off[chan,:])
        ds_off[chan,:] -= np.mean(ds_off[chan,:])
        if np.std(ds_off[chan,:])!=0:
            ds[chan,:] /= np.std(ds_off[chan,:])
        else: 
            ds[chan,:] = 0
    return ds

def SN(ts, t_cent, t_sig):
    ts_off = np.concatenate((ts[0:int(t_cent-3*t_sig)],ts[int(t_cent+3*t_sig):]))
    ts -= np.mean(ts_off)
    ts_off -= np.mean(ts_off)
    ts /= np.std(ts_off)
    ts_off /= np.std(ts_off)
    
    return ts


def plot(ds, extent, tsamp, plot_grid_idx, fig, width, t_cent, f_cent, t_sig, f_sig, index, burst_n, nrows, ncols, nbursts, plot_spectrum=False, colour='cyan',t_cent_2=None,t_sig_2=None,f_cent_2=None,f_sig_2=None):
    """
    Creates the family burst plot
    """
    freq_bottom = extent[2]
    conv = 2.355 #conversion from sigma to FWWHM
    
    ds_norm = normalise(ds,t_cent,t_sig)
    spectrum = np.mean(ds_norm[:,int(np.ceil(t_cent-2.*t_sig/conv)):int(np.ceil(t_cent+2.*t_sig/conv))], axis=1)
    res_f = (extent[3]-extent[2])/spectrum.size #frequency resolution

    if t_cent_2:
        t_cent_1 = np.min([t_cent,t_cent_2])
        if t_cent_2 == t_cent_1:
            t_cent_2 = t_cent
            t_sig_1 = t_sig_2
            t_sig_2 = t_sig
        else:
            t_sig_1 = t_sig

        t_cent = (t_cent_2-t_cent_1)/2. + t_cent_1

    if f_cent_2:
        f_cent_1 = np.min([f_cent,f_cent_2])
        if f_cent_2 == f_cent_1:
            f_cent_2 = f_cent
            f_sig_1 = f_sig_2
            f_sig_2 = f_sig
        else:
            f_sig_1 = f_sig
        f_cent = (f_cent_2-f_cent_1)/2.+ f_cent_1

    if f_cent and f_sig:
        f_l_bin = int(np.ceil((f_cent-2.*f_sig)))# - extent[2])/res_f))
        if f_sig_2:
            f_l_bin = int(np.ceil((f_cent_1-2.*f_sig_1)))
            
        if f_l_bin < 0:
            f_l_bin = 0
        f_h_bin = int(np.ceil((f_cent+2.*f_sig)))# - extent[2])/res_f))
        if f_sig_2:
            f_h_bin = int(np.ceil((f_cent_2+2.*f_sig_2)))
        if f_h_bin > spectrum.size:
            f_h_bin = spectrum.size
    else:
        f_l_bin = 0
        f_h_bin = spectrum.size


    bw = (f_h_bin - f_l_bin)*res_f #MHz
    
    peak = np.int(np.ceil(t_cent))
    extent[0] = - width / 2. # width is time window around the burst
    extent[1] = width / 2.

    t_l_bin = int(t_cent - (width/2.)/(tsamp*1e3))
    t_h_bin = int(t_cent + (width/2.)/(tsamp*1e3))
    if t_l_bin < 0 or t_h_bin > ds_norm.shape[1]:
        ds_norm = np.roll(ds_norm, int(ds_norm.shape[1]/2.-t_cent), axis=1)
        t_cent = int(ds_norm.shape[1]/2.)
        t_h_bin = int(t_cent + (width/2.)/(tsamp*1e3))
        t_l_bin = int(t_cent - (width/2.)/(tsamp*1e3))
    """
    if t_h_bin > ds_norm.shape[1]:
        ds_norm = np.roll(ds_norm, int(ds_norm.shape[1]/2.-t_cent), axis=1)
        t_cent = int(ds_norm.shape[1]/2.)
        t_h_bin = int(t_cent + (width/2.)/(tsamp*1e3))
        t_l_bin = int(t_cent - (width/2.)/(tsamp*1e3))
        #extent[1] = width/2. - (ds_norm.shape[1] - int(t_cent + (width/2.)/(tsamp*1e3)))*tsamp*1e3 
    """

    ts = np.mean(ds_norm[f_l_bin:f_h_bin,:], axis=0) # time series (summed the frequencies) around the burst
    ts = SN(ts, t_cent, t_sig)
    ds_norm = ds_norm[:,t_l_bin:t_h_bin]
    ts = ts[t_l_bin:t_h_bin]
    x = (np.arange(t_l_bin,t_h_bin,1)-int(t_cent))*tsamp*1e3
    


    rows = 2
    cols = 1
    if plot_spectrum: cols += 1
    plot_grid = gridspec.GridSpecFromSubplotSpec(rows, cols, plot_grid_idx, wspace=0., hspace=0.,height_ratios=[1,]*(rows-1)+[2,], width_ratios=[5,]+[1,]*(cols-1))
    ax1 = plt.Subplot(fig, plot_grid[rows-1,0])
    ax2 = plt.Subplot(fig, plot_grid[rows-2,0], sharex=ax1)
    if plot_spectrum: ax3 = plt.Subplot(fig, plot_grid[rows-1,1], sharey=ax1)
    units = ("GHz", "ms")
    
    cm1 = mpl.colors.ListedColormap(['black','red'])
    vmin = np.amin(ds_norm) #min of the array
    vmax = np.amax(ds_norm)   #max of the array
    zapped = np.where(spectrum==0) #identifying the frequencies that were zapped using pazi
    cmap = plt.cm.gist_yarg
    cmap.set_bad((1, 0, 0, 1)) #set color for masked values
    zap_size = int(ds_norm.shape[1]/18)
    ds_norm[zapped,:zap_size] = vmin-1000
    mask1=ds_norm<vmin-600
    mask1 = np.ma.masked_where(mask1==False,mask1)
    cmap.set_over(color='white') #set color used for high out-of-range values (the zapped values have been set to NaN)
    ds_norm[zapped,zap_size:] = vmax+1000.

    ax1.imshow(ds_norm, cmap=cmap, origin='lower', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax, extent=extent)
    ax1.imshow(mask1, cmap=cm1, origin='lower', aspect='auto', interpolation='nearest', vmin=0, vmax=1, extent=extent)
    ax1.set_xlim(extent[0]-0.0001, extent[1]+0.0001)
    ax1.set_ylim(extent[2],extent[3])

    #Label only edge plots
    if index % ncols == 0:
        ax1.set_ylabel(r'${\rm Frequency}\ ({\rm MHz})$')#.format(units[0]))
        ax2.set_ylabel('S/N')
        #ax1.set_yticklabels([r'$1.65$',r'$1.70$'])
    else:ax1.tick_params(axis='y', labelleft='off')
    ax1.tick_params(axis='x', labelbottom='off')

    excess = nrows*ncols - nbursts
    col = 0
    
    while excess > nrows:
        excess-=ncols
        col +=1
    
    indices = np.arange(1,nbursts+1,1)
    ind=(nrows-1-col)*ncols

    
    for i in range(ncols):
        
        if index ==indices[ind-excess+i-1]:
            ax1.tick_params(axis='x', labelbottom='on')
            ax1.set_xlabel(r'${\rm Time}\ ({\rm ms})$')



    #if (index <2) and width: ax1.tick_params(axis='x', labelbottom='off')
    #else:
    #    ax1.set_xlabel(r'${\rm Time}\ ({\rm ms})$')#.format(units[1]))
    ax1.xaxis.set_minor_locator(MultipleLocator(5))
    
    #plot pulse profile (time series)
    ax2.plot(x, ts, 'k-',alpha=1.0,zorder=1,lw=0.5,linestyle='steps-mid')
    point = ax2.scatter(x[0], ts[0], facecolors='none', edgecolors='none')
    
    #ax2.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    ax2.tick_params(axis='x', labelbottom='off', top='off')
    y_range = ts.max() - ts.min()
    ax2.set_ylim(-y_range/3., ts.max()*1.1)
    maxSN = np.ceil(np.max(ts))
    yticks=np.array([0,maxSN/2.,maxSN])
    ax2.set_yticks(yticks)


    if t_sig_2:
        ax2.hlines(y=-y_range/3.,xmin=(-(t_cent-t_cent_1+t_sig_1*2)*(tsamp*1e3)),xmax=((t_cent_2-t_cent+t_sig_2*2)*(tsamp*1e3)), lw=10,color=colour,zorder=0.8, alpha=0.2)
        ax2.hlines(y=-y_range/3.,xmin=(-(t_cent-t_cent_1+t_sig_1)*(tsamp*1e3)),xmax=((t_cent_2-t_cent+t_sig_2)*(tsamp*1e3)), lw=10,color=colour,zorder=0.8, alpha=0.4)
    else:
        ax2.hlines(y=-y_range/3.,xmin=(-t_sig*2*(tsamp*1e3)),xmax=(t_sig*2*(tsamp*1e3)), lw=10,color=colour,zorder=0.8, alpha=0.2)
        ax2.hlines(y=-y_range/3.,xmin=(-t_sig*(tsamp*1e3)),xmax=(t_sig*(tsamp*1e3)), lw=10,color=colour,zorder=0.8, alpha=0.4)
    
    b = np.argmin(np.abs(x-(-t_sig*2*(tsamp*1e3))))
    e = np.argmin(np.abs(x-(t_sig*2*(tsamp*1e3))))
    if t_sig_2:
        b = np.argmin(np.abs(x-(-((t_cent-t_cent_1)+t_sig_1*2)*(tsamp*1e3))))
        e = np.argmin(np.abs(x-((t_cent_2-t_cent+t_sig_2*2)*(tsamp*1e3))))

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width*0.65, box.height])
    legend1=ax2.legend((point,point), ((r"${\rm B%s}$")%burst_n,""), loc='upper left',handlelength=0,bbox_to_anchor=(0.01, 1.05), handletextpad=-0.5,frameon=False,markerscale=0, fontsize = 6)
    ax2.add_artist(legend1)
    ax2.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=False)

    legend2=ax2.legend((point,point), ((u"{} \u03bcs").format(int(tsamp*1e6)),""), loc='upper left',handlelength=0,bbox_to_anchor=(0.6, 1.05), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=6)
    ax2.add_artist(legend2)
    
    legend3=ax2.legend((point,point), ((u"{} MHz").format(int(res_f)),""), loc='upper left',handlelength=0,bbox_to_anchor=(0.6, 0.9), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=6)
    ax2.add_artist(legend3)

    ax2.tick_params(axis='y', which='major', pad=1.5)

    #plot spectrum (amplitude vs freq) only if plot_spectrum=True
    if plot_spectrum:
        y = np.linspace(extent[2], extent[3], spectrum.size)
        ax3.plot(spectrum, y, 'k-',zorder=2, lw=0.7,linestyle='steps-mid')
        ax3.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
        ax3.tick_params(axis='y', labelleft='off')
        ax3.set_ylim(extent[2],extent[3])
        x_range = spectrum.max() - spectrum.min()
        ax3.set_xlim(-x_range/3., x_range*6./5.)
        if f_sig_2:
            ax3.vlines(x=-x_range/3.,ymin=(extent[2]+(f_cent_1-2.*f_sig_1)*(res_f)),ymax=(extent[2]+(f_cent_2+2.*f_sig_2)*(res_f)), lw=20,color=colour,zorder=0.8, alpha=0.2)
            ax3.vlines(x=-x_range/3.,ymin=(extent[2]+(f_cent_1-f_sig_1)*(res_f)),ymax=(extent[2]+(f_cent_2+f_sig_2)*(res_f)), lw=20,color=colour,zorder=0.8, alpha=0.4)
            ax3.axhline(y=(extent[2]+(f_cent)*(res_f)),color=colour,lw=0.5)
        else:
            ax3.vlines(x=-x_range/3.,ymin=(extent[2]+(f_cent-2.*f_sig)*(res_f)),ymax=(extent[2]+(f_cent+2.*f_sig)*(res_f)), lw=20,color=colour,zorder=0.8, alpha=0.2)
            ax3.vlines(x=-x_range/3.,ymin=(extent[2]+(f_cent-f_sig)*(res_f)),ymax=(extent[2]+(f_cent+f_sig)*(res_f)), lw=20,color=colour,zorder=0.8, alpha=0.4)
            ax3.axhline(y=(extent[2]+(f_cent)*(res_f)),color=colour,lw=0.5)

    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
    if plot_spectrum: fig.add_subplot(ax3)
    
    return ts[b:e], bw 



if __name__ == '__main__':
    #adapt the figure parameters as needed:
    nrows = 3
    ncols = 6 
    width = 40. #Time window around the burst in ms.
    plot_grid = gridspec.GridSpec(nrows, ncols, wspace=0.2, hspace=0.1) #grid of burst plots
    fig = plt.figure(figsize=[8,7]) #defines the size of the plot
    archive_path = '/data1/nimmo/PRECISE/bursts/R67/VLBI/' #directory containing burst archives (create using dspsr)
    IDs_ordered = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    #IDs_ordered: dictionary Keys ordered by TOA.
    dm=412
    SEFD = 24./1.54
    dist = 453.

    #Dictionary of bursts 
    bursts = {"1": {'archive': '210410/StokesI_archives/pr153a_89_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 1013.125/2.,'f_cent': 78.6875/2., 't_sig':46.5/2., 'f_sig':66.91/2.,'f':16, 'b':16,'burst_n':89, 'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "2": {'archive': '210410/StokesI_archives/pr153a_85_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 2051.15625/2., 'f_cent': 122.77749999999992/2., 't_sig': 50.1478525/2., 'f_sig': 93.2488071/2., 'f':16, 'b':16, 'burst_n':85,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "3": {'archive': '210410/StokesI_archives/pr153a_5_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 1063.609375/2., 'f_cent': 56.99374999999998, 't_sig': 18.936053046875003/2., 'f_sig': 42.35179525, 'f':16, 'b':32, 'burst_n':5,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "4": {'archive': '210410/StokesI_archives/pr153a_47_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 235.78125/2., 'f_cent': 38.04875000000004, 't_sig': 21.829067500000004/2., 'f_sig': 29.1704402, 'f':16, 'b':32, 'burst_n':47,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "5": {'archive': '210410/StokesI_archives/pr153a_33_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 2025.953125/2., 'f_cent': 43.06875/2., 't_sig': 28.348117109375004/2., 'f_sig': 29.5464615/2., 'f':32, 'b':32, 'burst_n':33,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "6": {'archive': '210410/StokesI_archives/pr153a_17_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 1839.3125/2., 'f_cent': 87.3875/2., 't_sig': 52.4161346875/2., 'f_sig': 72.1935173/2., 'f':16, 'b':16, 'burst_n': 17,'colour':'darkorchid' ,'t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "7": {'archive': '210410/StokesI_archives/pr153a_79_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 370.4765625/2., 'f_cent': 16.63874999999996/2., 't_sig': 25.350899140625003/2., 'f_sig':25.9527508/2., 'f':32, 'b':32, 'burst_n':79,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "8": {'archive': '210410/StokesI_archives/pr153a_87_8us_125kHz_I_DM413.cor_Ef.ar.pazi' , 't_cent': 280.060546875, 'f_cent': 11.311874999999986, 't_sig': 8.64128251953125, 'f_sig': 24.1078892, 'f':32, 'b':64, 'burst_n':87,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "9": {'archive': '210410/StokesI_archives/pr153a_75_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 686.1796875/2., 'f_cent': 31.527499999999918/2., 't_sig': 27.620129062500002/2., 'f_sig': 48.7370501/2., 'f':32, 'b': 32, 'burst_n':75,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "10": {'archive': '210410/StokesI_archives/pr153a_73_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 485.6953125, 'f_cent': 42.54875, 't_sig':29.336086015625, 'f_sig':26.91552525, 'f':16, 'b':16, 'burst_n':73,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "11":{'archive': '210410/StokesI_archives/pr153a_65_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 1093.390625/2., 'f_cent': 78.1275/2., 't_sig': 22.071870625/2., 'f_sig': 64.4125633/2., 'f':16, 'b':16, 'burst_n':65,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "12":{'archive': '210410/StokesI_archives/pr153a_63_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent': 290.765625/2., 'f_cent': 116.5775/2., 't_sig': 75.70744562499999/2., 'f_sig': 78.9276972/2., 'f':16, 'b':16, 'burst_n':63,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "13":{'archive': '210410/StokesI_archives/pr153a_25_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent':421.1484375/2., 'f_cent':37.54875/2., 't_sig':21.822136484375005/2., 'f_sig':32.15761385/2., 'f':32, 'b':32, 'burst_n':25,'colour':'darkorchid','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "14":{'archive': '210419/StokesI_archives/pr156a_55_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent':556.234375/2., 'f_cent':153.0574999999999/2., 't_sig':15.199900000000003/2., 'f_sig':82.0784945/2., 'f':16, 'b':16, 'burst_n':55, 'colour':'mediumturquoise','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "15":{'archive': '210419/StokesI_archives/pr156a_82_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent':1239.734375/2., 'f_cent':113.3875/2., 't_sig':44.6125459375/2., 'f_sig':75.3342370/2., 'f':16, 'b':16, 'burst_n':82, 'colour':'mediumturquoise','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "16":{'archive': '210419/StokesI_archives/pr156a_97_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent':500.0859375/2., 'f_cent':111.0474999999999/4., 't_sig':57.13418593750001/4., 'f_sig':86.246576/4., 'f':32, 'b':32, 'burst_n':97, 'colour':'mediumturquoise','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "17":{'archive': '210419/StokesI_archives/pr156a_127_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent':658.5625/2., 'f_cent':77.84375/2., 't_sig':18.166188281249998/2., 'f_sig':51.2938805/2., 'f':32, 'b':32, 'burst_n':127, 'colour':'mediumturquoise','t_cent_2':None,'f_cent_2':None,'t_sig_2':None,'f_sig_2':None},
              "18":{'archive': '210419/StokesI_archives/pr156a_49_8us_125kHz_I_DM413.cor_Ef.ar.pazi', 't_cent':1304.53125/2., 'f_cent':140.75749999999994/2., 't_sig':33.64646671875/2., 'f_sig':52.1660901/2., 'f':16, 'b':16, 'burst_n':49, 'colour':'mediumturquoise','t_cent_2':1397.1875/2.,'f_cent_2':62.27749999999992/2.,'t_sig_2':21.20673125/2.,'f_sig_2':77.9218134/2.}
              }

    idx = 0
    for burst in IDs_ordered:
        burst=str(burst)
        ds, extent, tsamp = load_archive(os.path.join(archive_path,bursts[burst]['archive']),dm=dm,remove_baseline=False,extent=True,tscrunch=bursts[burst]['b'],fscrunch=bursts[burst]['f'])
        ts,bw = plot(ds,extent,tsamp,plot_grid[idx],fig,width,bursts[burst]['t_cent'],bursts[burst]['f_cent'],bursts[burst]['t_sig'],bursts[burst]['f_sig'],idx,bursts[burst]['burst_n'],nrows=nrows,ncols=ncols,nbursts=len(IDs_ordered),plot_spectrum=True,colour=bursts[burst]['colour'],t_cent_2=bursts[burst]['t_cent_2'],t_sig_2=bursts[burst]['t_sig_2'],f_cent_2=bursts[burst]['f_cent_2'],f_sig_2=bursts[burst]['f_sig_2'])
        print("--------------------")
        print("B"+str(bursts[burst]['burst_n']))
        print("--------------------")
        fluences(ts,tsamp,bw,SEFD,distance=dist)

        idx+=1

    
    fig.subplots_adjust(hspace=0.1, wspace=0.05, left=0.09,right=.96,bottom=.1,top=.99)
    plt.show()
    fig.savefig("burst_family.pdf", format = 'pdf', dpi=300)
