import filterbank
filterbank.__file__
import numpy as np
import matplotlib.pyplot as plt
import optparse
import rfifind
import matplotlib.cm
import matplotlib.gridspec as gridspec
import spectra
from numpy import array
import psrfits
import psr_utils
import sys
import os.path

def get_mask(rfimask, startsamp, N):
    """Return an array of boolean values to act as a mask
        for a Spectra object.

        Inputs:
            rfimask: An rfifind.rfifind object
            startsamp: Starting sample
            N: number of samples to read

        Output:
            mask: 2D numpy array of boolean values.
                True represents an element that should be masked.
    """
    sampnums = np.arange(startsamp, startsamp+N)
    blocknums = np.floor(sampnums/rfimask.ptsperint).astype('int')
    mask = np.zeros((N, rfimask.nchan), dtype='bool')
    for blocknum in np.unique(blocknums):
        blockmask = np.zeros_like(mask[blocknums==blocknum])
        chans_to_mask = rfimask.mask_zap_chans_per_int[blocknum]
        if chans_to_mask.any():
            blockmask[:,chans_to_mask] = True
        mask[blocknums==blocknum] = blockmask
    return mask.T

def maskfile(maskfn, data, start_bin, nbinsextra, extra_begin_chan=None, extra_end_chan=None):
    rfimask = rfifind.rfifind(maskfn)
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    num_chan = mask.shape[0]
    if extra_begin_chan!=None and extra_end_chan!=None:
        if len(extra_begin_chan) != len(extra_end_chan):
            print("length of extra mask begin channels and extra mask end channels need to be the same")
            sys.exit()
        else:
            for ms in range(len(extra_begin_chan)):
                end=num_chan-extra_begin_chan[ms]
                begin = num_chan-extra_end_chan[ms]
                mask[begin:end+1,:]=True

    masked_chans = mask.all(axis=1)
    # Mask data
    data = data.masked(mask, maskval='median-mid80')

    #datacopy = copy.deepcopy(data)
    return data, masked_chans

def waterfall(filename,start,duration,dm=0,mask=False,maskfn=None,favg=1,tavg=1,scaleindep=False,extra_begin_chan=None,extra_end_chan=None):
    """

    """
    if filename.endswith('.fil'):
        rawdatafile = filterbank.filterbank(filename)
        tsamp=rawdatafile.header['tsamp']
        nchans=rawdatafile.header['nchans']
        freqs=rawdatafile.frequencies
        total_N=rawdatafile.number_of_samples
        df = np.abs(freqs[-1] - freqs[0])
        fres=df/int(nchans+1)
        scan_start=rawdatafile.tstart

    if filename.endswith('.fits'):
        rawdatafile = psrfits.PsrfitsFile(filename)
        tsamp=rawdatafile.tsamp
        nchans=rawdatafile.nchan
        freqs=rawdatafile.frequencies
        total_N=rawdatafile.specinfo.N
        df =  np.abs(freqs[-1] - freqs[0])
        fres = df/int(nchans+1)
        scan_start=rawdatafile.header['STT_IMJD']+(rawdatafile.header['STT_SMJD']+rawdatafile.header['STT_OFFS'])/(24.*3600.)

    start_bin=np.round(start/tsamp).astype('int')    #convert begin time to bins
    dmdelay_coeff = 4.15e3 * np.abs(1./freqs[0]**2 - 1./freqs[-1]**2)
    nbins = np.round(duration/tsamp).astype('int')     #convert duration to bins

    if dm!=0:
        nbinsextra = np.round((duration + dmdelay_coeff * dm)/tsamp).astype('int')
    else:
        nbinsextra = nbins

    # If at end of observation
    if (start_bin + nbinsextra) > total_N-1:
        nbinsextra = total_N-1-start_bin

    data = rawdatafile.get_spectra(start_bin, nbinsextra)

    #masking
    if mask and maskfn:
        data, masked_chans = maskfile(maskfn, data, start_bin, nbinsextra,extra_begin_chan=extra_begin_chan, extra_end_chan=extra_end_chan)
    else:
        masked_chans = np.zeros(nchans,dtype=bool)

    data_masked = np.ma.masked_array(data.data)
    data_masked[masked_chans] = np.ma.masked
    data.data = data_masked

    if dm!=0:
        data.dedisperse(dm, padval='mean')

    data.downsample(tavg)

    # scale data
    data_noscale=data
    data = data.scaled(scaleindep)

    return data, data_noscale, nbinsextra, nbins, start,tsamp,fres,scan_start

def plot_waterfall(data,data_noscale,start,duration,centre_MJD,dm,sigma,width,tsamp,\
                   fres,tavg=1,favg=1,cmap_str="gist_yarg",integrate_spec=False,sweep_dms=[], sweep_posns=[],save_plot=False,interactive=True):

    # Set up axes
    if interactive:
        fig = plt.figure(figsize=[12,7])

    fig.text(0.1,0.9,"DM = %s pc/cc"%dm)
    fig.text(0.1,0.85, "Centre MJD = %s"%centre_MJD)
    fig.text(0.1,0.8,"Significance = %s sigma"%sigma)
    fig.text(0.1,0.75,"Boxcar width = %s seconds"%(tsamp*width))
    fig.text(0.1,0.25, "Plotting resolution")
    fig.text(0.1,0.2,"Time resolution %s milliseconds"%(tsamp*tavg*1e3))
    fig.text(0.1,0.15,"Frequency resolution %s MHz"%(fres*favg))

    im_width = 0.3 if integrate_spec else 0.4
    im_height = 0.8
    ax_im = plt.axes((0.45, 0.15, im_width, im_height-0.2))
    ax_ts = plt.axes((0.45, 0.75, im_width, 0.2),sharex=ax_im)
    if integrate_spec: ax_spec = plt.axes((0.75, 0.15, 0.2, im_height),sharey=ax_im)

    # Ploting it up
    nbinlim = np.int(duration/data.dt)

    #downsample in frequency
    if favg!=None and favg>1:
        nchan_tot = data.data.shape[0]
        favg=float(favg)
        if (nchan_tot/favg)-int(nchan_tot/favg)!=0:
            print("The total number of channels is %s, please choose an fscrunch val\
ue that divides the total number of channels."%nchan_tot)
            sys.exit()
        else:
            newnchan=nchan_tot/favg
            data.data=np.array(np.row_stack([np.mean(subint, axis=0) for subint in np.vsplit(data.data,newnchan)]))

    img = ax_im.imshow(data.data[..., :nbinlim], aspect='auto', \
                cmap=matplotlib.cm.cmap_d[cmap_str], \
                interpolation='nearest', origin='upper', \
                extent=(data.starttime, data.starttime+ nbinlim*data.dt, \
                        data.freqs.min(), data.freqs.max()))

    # Sweeping it up
    for ii, sweep_dm in enumerate(sweep_dms):
        ddm = sweep_dm-data.dm
        delays = psr_utils.delay_from_DM(ddm, data.freqs)
        delays -= delays.min()

        if sweep_posns is None:
            sweep_posn = 0.0
        elif len(sweep_posns) == 1:
            sweep_posn = sweep_posns[0]
        else:
            sweep_posn = sweep_posns[ii]
        sweepstart = data.dt*data.numspectra*sweep_posn+data.starttime
        #sty = SWEEP_STYLES[ii%len(SWEEP_STYLES)]
        ax_im.plot(delays+sweepstart, data.freqs, lw=4, alpha=0.5)

    # Dressing it up
    ax_im.xaxis.get_major_formatter().set_useOffset(False)
    ax_im.set_xlabel("Time")
    ax_im.set_ylabel("Frequency (MHz)")
    ax_im.set_ylim((data.freqs.min(), data.freqs.max()))
    
    
    axsec = ax_im.twinx()

    axsec.set_yticks(np.round(np.linspace(0,data.data[..., :nbinlim].shape[0],10))*favg)
    axsec.set_ylabel('Bins')

    # Plot Time series

    Data = np.array(data_noscale.data[..., :nbinlim])
    Dedisp_ts = Data.sum(axis=0)
    times = (np.arange(data_noscale.numspectra)*data_noscale.dt + start)[..., :nbinlim]

    ax_ts.plot(times, Dedisp_ts,"k")
    ax_ts.set_xlim([times.min(),times.max()])
    plt.setp(ax_ts.get_xticklabels(), visible = False)
    plt.setp(ax_ts.get_yticklabels(), visible = False)

    # Plot Spectrum
    if integrate_spec:
        spectrum_window = 0.05*duration
        window_width = int(spectrum_window/data.dt) # bins
        burst_bin = nbinlim/2
        on_spec = np.array(data.data[..., burst_bin-window_width:burst_bin+window_width])
        Dedisp_spec = on_spec.sum(axis=1)[::-1]

        freqs = np.linspace(data.freqs.min(), data.freqs.max(), len(Dedisp_spec))
        ax_spec.plot(Dedisp_spec,freqs,"k")
        plt.setp(ax_spec.get_xticklabels(), visible = False)
        plt.setp(ax_spec.get_yticklabels(), visible = False)
        ax_spec.set_ylim([data.freqs.min(),data.freqs.max()])

        ax_ts.axvline(times[burst_bin]-spectrum_window,ls="--",c="grey")
        ax_ts.axvline(times[burst_bin]+spectrum_window,ls="--",c="grey")

    if interactive:
        fig.canvas.mpl_connect('key_press_event', \
                lambda ev: (ev.key in ('q','Q') and plt.close(fig)))

    if save_plot==True:
        fig.savefig('waterfall_%sMJD_DM%s_sigma%s.png'%(centre_MJD,dm,sigma),dpi=300,facecolor='w', edgecolor='w',format='png')
        plt.close()
    else:
        plt.show()



def main():
    filename=args[0]

    if os.path.isfile('./pulse_cands/waterfall_*MJD_DM%s_sigma%s.png'%(options.dm,options.sigma)) == False:
 
        data, data_noscale, bins, nbins, start,tsamp,fres, scan_start = waterfall(filename, options.start, \
                            options.duration, dm=options.dm,\
                            mask=options.mask, maskfn=options.maskfile, \
                            favg=options.favg,tavg=options.tavg, \
                            scaleindep=options.scaleindep, extra_begin_chan=options.begin_mask, extra_end_chan = options.end_mask)


        centre_MJD=scan_start+((options.start + (0.5*options.duration))/(24.*3600.))

        plot_waterfall(data,data_noscale,start,options.duration,centre_MJD,options.dm,options.sigma,options.width,tsamp,\
                   fres,tavg=options.tavg,favg=options.favg,cmap_str=options.cmap,integrate_spec=options.integrate_spec,sweep_dms=options.sweep_dms, sweep_posns=options.sweep_posns,save_plot=options.save_plot)




if __name__=='__main__':
    parser = optparse.OptionParser(prog="waterfall_plot.py", \
                        usage="%prog [OPTIONS] INFILE", \
                        description="Create a waterfall plot to show the " \
                                    "frequency sweep of a single pulse " \
                                    "in filterbank data.")
    parser.add_option('-d', '--dm', dest='dm', type='float', \
                        help="DM to use when dedispersing data for plot. " \
                                "(Default: 0 pc/cm^3)", default=0.0)
    parser.add_option('-T', '--start-time', dest='start', type='float', \
                        help="Time into observation (in seconds) at which " \
                                "to start plot.")
    parser.add_option('--show-spec', dest='integrate_spec', action='store_true', \
                        help="Plot the spectrum. " \
                                "(Default: Do not show the spectrum)", default=False)
    parser.add_option('-t', '--duration', dest='duration', type='float', \
                        help="Duration (in seconds) of plot.")
    parser.add_option('--sigma', dest='sigma', type='float', \
                        help="Significance from single_pulse_search.py")
    parser.add_option('--width', dest='width', type='int', \
                        help="Boxcar width in bins from single_pulse_search.py", default=0)
    parser.add_option('--sweep-dm', dest='sweep_dms', type='float', \
                        action='append', \
                        help="Show the frequency sweep using this DM. " \
                                "(Default: Don't show sweep)", default=[])
    parser.add_option('--sweep-posn', dest='sweep_posns', type='float', \
                        action='append', \
                        help="Show the frequency sweep at this position. " \
                                "The position refers to the high-frequency " \
                                "edge of the plot. Also, the position should " \
                                "be a number between 0 and 1, where 0 is the " \
                                "left edge of the plot. "
                                "(Default: 0)", default=None)
    parser.add_option('--tavg', dest='tavg', type='int', \
                        help="Factor to downsample data by (time). (Default: 1).", \
                        default=1)
    parser.add_option('--favg', dest='favg', type='int', \
                        help="Factor to downsample data by (frequency). Must be a divisor of the number of channels in the data. (Default: 1).", \
                        default=1)
    parser.add_option('--maskfile', dest='maskfile', type='string', \
                        help="Mask file produced by rfifind. Used for " \
                             "masking and bandpass correction.", \
                        default=None)
    parser.add_option('--mask', dest='mask', action="store_true", \
                        help="Mask data using rfifind mask (Default: Don't mask).", \
                        default=False)
    parser.add_option('--scaleindep', dest='scaleindep', action='store_true', \
                        help="If this flag is set scale each channel " \
                                "independently. (Default: Scale using " \
                                "global maximum.)", \
                        default=False)
    parser.add_option('--save_plot', dest='save_plot', action='store_true', \
                        help="If this is set, it saves the waterfall plots " \
                                "as png. (Default: plot on screen)",\
                        default=False)
    parser.add_option('--colour-map', dest='cmap', \
                        help="The name of a valid matplotlib colour map." \
                                "(Default: gist_yarg.)", \
                        default='gist_yarg')
    parser.add_option('--begin_mask', dest='begin_mask', type='string', \
                        help="Begin channel for additional masking. If multiple frequency regions list with comma e.g. --begin_mask 60,100,345", default=None)
    parser.add_option('--end_mask', dest='end_mask', type='string', \
                        help="End channel for additional masking", default=None)
                        
    options, args = parser.parse_args()

    if not hasattr(options, 'start'):
        raise ValueError("Start time (-T/--start-time) " \
                            "must be given on command line!")
    if not hasattr(options, 'duration'):
        raise ValueError("Duration (-t/--duration) " \
                            "must be given on command line!")
    if options.begin_mask!=None:
        if ',' not in options.begin_mask:
            options.begin_mask = [int(options.begin_mask)]
            options.end_mask = [int(options.end_mask)]
        else:
            options.begin_mask = [int(x.strip()) for x in options.begin_mask.split(',')]
            options.end_mask = [int(x.strip()) for x in options.end_mask.split(',')]

    main()
