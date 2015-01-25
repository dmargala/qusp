#!/usr/bin/env python
"""
"""
import argparse

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 8})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from sklearn import linear_model
import scipy.optimize

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def mad(arr):
    """
    Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def nmad(arr):
    return 1.4826*mad(arr)

def add_stat_legend(x, prec=2):
    textstr = ''
    textstr += '$\mathrm{N}=%d$\n' % len(x)
    textstr += ('$\mathrm{mean}=%.'+str(prec)+'g$\n') % np.nanmean(x)
    textstr += ('$\mathrm{median}=%.'+str(prec)+'g$\n') % np.nanmedian(x)
    textstr += ('$\mathrm{std}=%.'+str(prec)+'g$\n') % np.nanstd(x)
    textstr += ('$\mathrm{nMAD}=%.'+str(prec)+'g$') % nmad(x)
    props = dict(boxstyle='round', facecolor='white')
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, 
        va='top', ha='right', bbox=props)

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-o", "--output", type=str, default=None,
        help="output file base name")
    parser.add_argument("-x", "--x-input", type=str, default=None,
        help="required input file")
    parser.add_argument("-y", "--y-input", type=str, default=None,
        help="required input file")
    parser.add_argument("--correct-x", action="store_true",
        help="correct SDSS mag to AB mag")
    args = parser.parse_args()

    # these are the bands we're using
    bands = 'gri'

    # trim filename to a useful label for plots later on
    xname = args.x_input.split('/')[-1].split('.')[0]
    yname = args.y_input.split('/')[-1].split('.')[0]
    print '%s vs %s' % (xname, yname)
    print

    # import data and filter missing entries
    xdata_raw = np.loadtxt(args.x_input)
    ydata_raw = np.loadtxt(args.y_input)

    # filter missing data
    mask = np.logical_not(np.any(xdata_raw == 0, axis=1)) & np.logical_not(np.any(ydata_raw == 0, axis=1)) #& (ydata_raw[:,0] < 19)
    
    xdata = xdata_raw[mask]
    ydata = ydata_raw[mask]

    print np.sum(mask)

    # if requested, convert sdss mags to ab mags on xdata
    if args.correct_x:
        xdata += [0.012, 0.010, 0.028]

    # mag scatter
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot2grid((1,3), (0,0))
    ax2 = plt.subplot2grid((1,3), (0,1))
    ax3 = plt.subplot2grid((1,3), (0,2))
    axs = [ax1, ax2, ax3]
    for i in range(len(axs)):
        plt.sca(axs[i])
        x = xdata[:,i]
        y = xdata[:,i]-ydata[:,i]
        # plt.hist2d(x, x-y, bins=50)#, 'o', alpha=.3, lw=0, ms=2)

        # Estimate the 2D histogram
        H, xedges, yedges = np.histogram2d(x,y,bins=(np.linspace(14,27,101), np.linspace(-1.5,1.5,101)))
         
        # H needs to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)
         
        # Mask zeros
        Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
         
        # Plot 2D histogram using pcolor
        plt.pcolormesh(xedges,yedges,Hmasked)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # cbar = plt.colorbar()
        # cbar.ax.set_ylabel('Counts')

        plt.ylim([-1.5,1.5])
        plt.xlim([14,27])
        plt.grid(True)
        plt.title(xname+' vs. '+yname)
        plt.xlabel(bands[i])
        plt.ylabel(r'$\Delta$%s'%bands[i])
        add_stat_legend(y, prec=3)
    fig.savefig(args.output+'-gri.png', bbox_inches='tight')

    # delta mag histograms
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot2grid((1,3), (0,0))
    ax2 = plt.subplot2grid((1,3), (0,1))
    ax3 = plt.subplot2grid((1,3), (0,2))
    axs = [ax1, ax2, ax3]
    colors = ['green','red','black']
    print 'delta mag (band: mean / rms / nmad):'
    for i in range(len(axs)):
        plt.sca(axs[i])
        x = xdata[:,i]
        y = ydata[:,i]
        xydiff = x-y
        print '  %s: %.4f / %.4f / %.4f' % (bands[i], np.mean(xydiff), np.sqrt(np.var(xydiff)), nmad(xydiff))
        plt.hist(xydiff, bins=np.linspace(-1,1,101,endpoint=True), histtype='step', color=colors[i])#, lw=0, alpha=.5)
        plt.grid(True)
        plt.xlim([-1,1])
        plt.xlabel('Synthetic Mag - PSF Mag')
        add_stat_legend(xydiff, prec=3)
    fig.savefig(args.output+'-hist.png', bbox_inches='tight')

    # color
    fig = plt.figure(figsize=(8,4))
    ax1 = plt.subplot2grid((1,2), (0,0))
    ax2 = plt.subplot2grid((1,2), (0,1))
    axs = [ax1, ax2]
    print 'delta color (color: mean / rms / nmad):'
    xaxiscol = 0
    for iax,(i,j) in enumerate([(0,1),(1,2)]):
        plt.sca(axs[iax])
        xcolor = xdata[:,i]-xdata[:,j]
        ycolor = ydata[:,i]-ydata[:,j]
        xydiff = xcolor-ycolor
        print '  %s-%s: %.4f / %.4f / %.4f' % (bands[i], bands[j], np.mean(xydiff), np.sqrt(np.var(xydiff)), nmad(xydiff))

        x = xdata[:,xaxiscol]
        y = xydiff
        #plt.hist2d(xdata[:,xaxiscol], xydiff, bins=50)#, 'o', alpha=.3, lw=0, ms=2)
        # Estimate the 2D histogram
        H, xedges, yedges = np.histogram2d(x,y,bins=(np.linspace(14,27,101), np.linspace(-1,1,101)))
         
        # H needs to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)
         
        # Mask zeros
        Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
         
        # Plot 2D histogram using pcolor
        plt.pcolormesh(xedges,yedges,Hmasked)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # cbar = plt.colorbar()
        # cbar.ax.set_ylabel('Counts')

        plt.ylim([-1,1])
        plt.xlim([14,27])
        plt.grid(True)

        plt.xlabel('%s'%(bands[xaxiscol]))
        plt.ylabel(r'$\Delta(%s-%s)$'%(bands[i],bands[j]))
        plt.title(xname+' vs. '+yname)
        add_stat_legend(y, prec=3)
    fig.savefig(args.output+'-colors.png', bbox_inches='tight')

    # delta color histograms
    fig = plt.figure(figsize=(8,4))
    ax1 = plt.subplot2grid((1,2), (0,0))
    ax2 = plt.subplot2grid((1,2), (0,1))
    axs = [ax1, ax2]
    xaxiscol = 1
    for iax,(i,j) in enumerate([(0,1),(1,2)]):
        plt.sca(axs[iax])
        xcolor = xdata[:,i]-xdata[:,j]
        ycolor = ydata[:,i]-ydata[:,j]
        xydiff = xcolor-ycolor
        plt.hist(xydiff, bins=np.linspace(-1,1,101,endpoint=True), histtype='step')#, lw=0, alpha=.5)
        plt.xlim([-1,1])
        plt.grid(True)
        plt.xlabel(r'$\Delta(%s-%s)$'%(bands[i],bands[j]))
        plt.title('Synthetic - PSF')
        add_stat_legend(xydiff, prec=3)
    fig.savefig(args.output+'-color-hist.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
