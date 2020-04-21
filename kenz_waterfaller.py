#!/usr/bin/env python

"""
waterfaller.py

Make waterfall plots to show frequency sweep of a single pulse.
Reads PSRFITS or SIGPROC filterbank format files.

Patrick Lazarus - Aug. 19, 2011
Paul Scholz - Nov 2015

Kenzie Nimmo - 2020

"""

import sys
import optparse
import copy

import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import matplotlib.gridspec as gridspec

import psr_utils
import rfifind

import psrfits
import filterbank
import spectra

from numpy import array


SWEEP_STYLES = ['r-', 'b-', 'g-', 'm-', 'c-']


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

def maskfile(maskfn, data, start_bin, nbinsextra):
    rfimask = rfifind.rfifind(maskfn)
    mask = get_mask(rfimask, start_bin, nbinsextra)[::-1]
    masked_chans = mask.all(axis=1)
    # Mask data
    data = data.masked(mask, maskval='median-mid80')

    #datacopy = copy.deepcopy(data)
    return data, masked_chans

def waterfall(rawdatafile, start, duration, dm=None, nbins=None, nsub=None,\
              subdm=None, zerodm=False, tavg=1, scaleindep=False,\
              width_bins=1, mask=False, maskfn=None, bandpass_corr=False, \
              ref_freq=None):
    """
    Create a waterfall plot (i.e. dynamic specrum) from a raw data file.
   Inputs:
       rawdatafile - a PsrfitsData instance.
       start - start time of the data to be read in for waterfalling.
       duration - duration of data to be waterfalled.
    Optional Inputs:
       dm - DM to use when dedispersing data.
             Default: Don't de-disperse
       nbins - Number of time bins to plot. This option overrides
                the duration argument.
                Default: determine nbins from duration.
       nsub - Number of subbands to use. Must be a factor of number of channels.
               Default: Number of channels.
       subdm - DM to use when subbanding. Default: same as dm argument.
       zerodm - subtract mean of each time-sample from data before
    de-dispersing.
    tavg - Factor to downsample in time by. Default: Don't downsample.
       scaleindep - Scale each channel independently.
                     Default: Scale using global maximum.
       width_bins - Smooth each channel/subband with a boxcar width_bins wide.
                     Default: Don't smooth.
       maskfn - Filename of RFIFIND mask to use for masking data.
                 Default: Don't mask data.
       bandpass_corr - Correct for the bandpass. Requires an rfifind
                        mask provided by maskfn keyword argument.
                        Default: Do not remove bandpass.
       ref_freq - Reference frequency to de-disperse to.
                   If subbanding and de-dispersing the start time
                   will be corrected to account for change in
                   reference frequency.
                   Default: Frequency of top channel.
    Outputs:
       data - Spectra instance of waterfalled data cube.
       nbinsextra - number of time bins read in from raw data.
       nbins - number of bins in duration.
       start - corrected start time.
    """

    if subdm is None:
        subdm = dm

    # Read data
    if ref_freq is None:
        ref_freq = rawdatafile.freqs.max()

        if nsub and dm:
            df = rawdatafile.freqs[1] - rawdatafile.freqs[0]
            nchan_per_sub = rawdatafile.nchan/nsub
            top_ctrfreq = rawdatafile.freqs.max() - \
                      0.5*nchan_per_sub*df # center of top subband
            start += 4.15e3 * np.abs(1./ref_freq**2 - 1./top_ctrfreq**2) * dm

    start_bin = np.round(start/rawdatafile.tsamp).astype('int')
    dmfac = 4.15e3 * np.abs(1./rawdatafile.frequencies[0]**2 - 1./rawdatafile.frequencies[-1]**2)

    if nbins is None:
        nbins = np.round(duration/rawdatafile.tsamp).astype('int')

    if dm:
        nbinsextra = np.round((duration + dmfac * dm)/rawdatafile.tsamp).astype('int')
    else:
        nbinsextra = nbins

    # If at end of observation
    if (start_bin + nbinsextra) > rawdatafile.nspec-1:
        nbinsextra = rawdatafile.nspec-1-start_bin
        
    data = rawdatafile.get_spectra(start_bin, nbinsextra)

    # Masking
    if mask and maskfn:
        data, masked_chans = maskfile(maskfn, data, start_bin, nbinsextra)
    else:
        masked_chans = np.zeros(rawdatafile.nchan,dtype=bool)

    # Bandpass correction
    if maskfn and bandpass_corr:
        bandpass = rfifind.rfifind(maskfn).bandpass_avg[::-1]
        #bandpass[bandpass == 0] = np.min(bandpass[np.nonzero(bandpass)])
        masked_chans[bandpass == 0] = True

        # ignore top and bottom 1% of band
        ignore_chans = np.ceil(0.01*rawdatafile.nchan)
        masked_chans[:ignore_chans] = True
        masked_chans[-ignore_chans:] = True


    data_masked = np.ma.masked_array(data.data)
    data_masked[masked_chans] = np.ma.masked
    data.data = data_masked

    if bandpass_corr:
       data.data /= bandpass[:, None]

    # Zerodm filtering
    if (zerodm == True):
        data.data -=  data.data.mean(axis=0)


    # Subband data
    if (nsub is not None) and (subdm is not None):
        data.subband(nsub, subdm, padval='mean')

    # Dedisperse
    if dm:
        data.dedisperse(dm, padval='mean')

    # Downsample
    data.downsample(tavg)


    # scale data
    data_noscale=data
    data = data.scaled(scaleindep)

    # Smooth
    if width_bins > 1:
        data.smooth(width_bins, padval='mean')

    return data, data_noscale, nbinsextra, nbins, start

def plot_waterfall(data,data_noscale, start, duration,centre_MJD,dm,sigma, width, tsamp,tavg,fres,favg,integrate_spec=False, show_cb=False,cmap_str="gist_yarg", sweep_dms=[], sweep_posns=[], interactive=True):
    
    
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
            print("The total number of channels is %s, please choose an fscrunch value that divides the total number of channels."%nchan_tot)
            sys.exit()
        else:
            newnchan=nchan_tot/favg
            data.data=np.array(np.row_stack([np.mean(subint, axis=0) for subint in np.vsplit(data.data,newnchan)]))






    img = ax_im.imshow(data.data[..., :nbinlim], aspect='auto', \
                cmap=matplotlib.cm.cmap_d[cmap_str], \
                interpolation='nearest', origin='upper', \
                extent=(data.starttime, data.starttime+ nbinlim*data.dt, \
                        data.freqs.min(), data.freqs.max()))
    if show_cb:
        cb = ax_im.get_figure().colorbar(img)
        cb.set_label("Scaled signal intensity (arbitrary units)")

    #plt.axis('tight')
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
        sty = SWEEP_STYLES[ii%len(SWEEP_STYLES)]
        ax_im.plot(delays+sweepstart, data.freqs, sty, lw=4, alpha=0.5)
        
    # Dressing it up
    ax_im.xaxis.get_major_formatter().set_useOffset(False)
    ax_im.set_xlabel("Time")
    ax_im.set_ylabel("Frequency (MHz)")
    ax_im.set_ylim((data.freqs.min(), data.freqs.max()))

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

        plt.show()

def main():
    fn = args[0]

    if fn.endswith(".fil"):
        # Filterbank file
        filetype = "filterbank"
        rawdatafile = filterbank.FilterbankFile(fn)
        scan_start=rawdatafile.tstart
        tsamp=rawdatafile.tsamp
        df = rawdatafile.freqs[1] - rawdatafile.freqs[0]
        fres=df/int(rawdatafile.nchan+1)
    elif fn.endswith(".fits"):
        # PSRFITS file
        filetype = "psrfits"
        rawdatafile = psrfits.PsrfitsFile(fn)
    else:
        raise ValueError("Cannot recognize data file type from "
                         "extension. (Only '.fits' and '.fil' "
                         "are supported.)")

    data, data_noscale, bins, nbins, start = waterfall(rawdatafile, options.start, \
                            options.duration, dm=options.dm,\
                            nbins=options.nbins, nsub=options.nsub,\
                            subdm=options.subdm, zerodm=options.zerodm, \
                            tavg=options.tavg, \
                            scaleindep=options.scaleindep, \
                            width_bins=options.width_bins, mask=options.mask, \
                            maskfn=options.maskfile, \
                            bandpass_corr=options.bandpass_corr)

    centre_MJD=scan_start+((options.start + (0.5*options.duration))/(24.*3600.))
    plot_waterfall(data, data_noscale, start, options.duration, centre_MJD, options.dm,\
                   options.sigma,options.width,tsamp,options.tavg,fres, options.favg, integrate_spec=options.integrate_spec, show_cb=options.show_cb,
                   cmap_str=options.cmap, sweep_dms=options.sweep_dms,
                   sweep_posns=options.sweep_posns)

if __name__=='__main__':
    parser = optparse.OptionParser(prog="waterfaller.py", \
                        version="v0.9 Patrick Lazarus (Aug. 19, 2011)", \
                        usage="%prog [OPTIONS] INFILE", \
                        description="Create a waterfall plot to show the " \
                                    "frequency sweep of a single pulse " \
                                    "in psrFits data.")
    parser.add_option('--subdm', dest='subdm', type='float', \
                        help="DM to use when subbanding. (Default: " \
                                "same as --dm)", default=None)
    parser.add_option('--zerodm', dest='zerodm', action='store_true', \
                        help="If this flag is set - Turn Zerodm filter - ON  (Default: " \
                                "OFF)", default=False)
    parser.add_option('-s', '--nsub', dest='nsub', type='int', \
                        help="Number of subbands to use. Must be a factor " \
                                "of number of channels. (Default: " \
                                "number of channels)", default=None)
    parser.add_option('-d', '--dm', dest='dm', type='float', \
                        help="DM to use when dedispersing data for plot. " \
                                "(Default: 0 pc/cm^3)", default=0.0)
    parser.add_option('--show-spec', dest='integrate_spec', action='store_true', \
                        help="Plot the spectrum. " \
                                "(Default: Do not show the spectrum)", default=False)
    parser.add_option('--bandpass', dest='bandpass_corr', action='store_true', \
                        help="Correct for the bandpass. Requires an rfifind " \
                                "mask provided by --mask option." \
                                "(Default: Do not remove bandpass)", default=False)
    parser.add_option('-T', '--start-time', dest='start', type='float', \
                        help="Time into observation (in seconds) at which " \
                                "to start plot.")
    parser.add_option('-t', '--duration', dest='duration', type='float', \
                        help="Duration (in seconds) of plot.")
    parser.add_option('-n', '--nbins', dest='nbins', type='int', \
                        help="Number of time bins to plot. This option takes " \
                                "precedence over -t/--duration if both are " \
                                "provided.")
    parser.add_option('--sigma', dest='sigma', type='float', \
                        help="Significance from single_pulse_search.py")
    parser.add_option('--width', dest='width', type='int', \
                        help="Boxcar width in bins from single_pulse_search.py", default=0)
    parser.add_option('--width-bins', dest='width_bins', type='int', \
                        help="Smooth each channel/subband with a boxcar " \
                                "this many bins wide. (Default: Don't smooth)", \
                        default=1)
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
    parser.add_option('--show-colour-bar', dest='show_cb', action='store_true', \
                        help="If this flag is set show a colour bar. " \
                                "(Default: No colour bar.)", \
                        default=False)
    parser.add_option('--colour-map', dest='cmap', \
                        help="The name of a valid matplotlib colour map." \
                                "(Default: gist_yarg.)", \
                        default='gist_yarg')
    options, args = parser.parse_args()

    if not hasattr(options, 'start'):
        raise ValueError("Start time (-T/--start-time) " \
                            "must be given on command line!")
    if (not hasattr(options, 'duration')) and (not hasattr(options, 'nbins')):
        raise ValueError("One of duration (-t/--duration) " \
                            "and num bins (-n/--nbins)" \
                            "must be given on command line!")
    if options.subdm is None:
        options.subdm = options.dm
    main()
