#!/usr/bin/env python
"""
"""
import argparse

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from sklearn import linear_model
import scipy.optimize
 
def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level
 
def density_contour(xdata, ydata, nbins_x, nbins_y, xlim, ylim, ax=None, **contour_kwargs):
    """ Create a density contour plot.

    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """
 
    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), range=[xlim, ylim], normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((nbins_x,1))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((1,nbins_y))
 
    pdf = (H*(x_bin_sizes*y_bin_sizes))
 
    one_sigma = scipy.optimize.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    two_sigma = scipy.optimize.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    #three_sigma = scipy.optimize.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    #levels = [one_sigma, two_sigma three_sigma]
    levels = [one_sigma]
 
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T
 
    if ax == None:
        contour = plt.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
    else:
        contour = ax.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
 
    return contour

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
    textstr += ('$\mathrm{NMAD}=%.'+str(prec)+'g$') % nmad(x)
    props = dict(boxstyle='round', facecolor='white')
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes, 
        va='top', ha='right', bbox=props)

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-o", "--output", type=str, default='mag-residuals-hist',
        help="output file base name")
    parser.add_argument("--sdss-mags", type=str, default=None,
        help="sdss imaging magnitudes")
    parser.add_argument("--boss-mags", type=str, default=None,
        help="boss synthetic magnitudes")
    parser.add_argument("--corrected-mags", type=str, default=None,
        help="corrected synthetic magnitudes")
    parser.add_argument("--anc-mags", type=str, default=None,
        help="ancillary reduction synthetic magnitudes")
    parser.add_argument("--correct-ab", action="store_true",
        help="correct SDSS mag to AB mag")
    args = parser.parse_args()

    # these are the bands we're using
    bands = 'gri'

    # import data and filter missing entries
    sdss_mags = np.loadtxt(args.sdss_mags)
    boss_mags = np.loadtxt(args.boss_mags)
    corrected_mags = np.loadtxt(args.corrected_mags)
    if args.anc_mags:
        ancillary_mags = np.loadtxt(args.anc_mags)

    def get_residuals(x, y):
        # filter missing data
        mask = np.logical_not(np.any(x == 0, axis=1)) & np.logical_not(np.any(y == 0, axis=1)) #& (ydata_raw[:,0] < 19)
        if args.verbose:
            print 'Number entries kept: %d' % np.sum(mask)
        x = x[mask]
        y = y[mask]
        # convert sdss mags to ab mags on xdata
        #if args.correct_ab:
        #    sdss_data += [0.012, 0.010, 0.028]
        return x-y, x, y

    res1, x1, y1 = get_residuals(sdss_mags, boss_mags)
    res2, x2, y2 = get_residuals(sdss_mags, corrected_mags)

    if args.anc_mags:
        res3, x3, y3 = get_residuals(sdss_mags, ancillary_mags)

    if args.verbose:
        print 'delta mag (band: mean & rms & nmad):'

    for i in [0, 1, 2]:
        if args.verbose:
            print '  %s:' % bands[i]
            print '%.3f & %.3f & %.3f' % (np.mean(res1[i]), np.sqrt(np.var(res1[i])), nmad(res1[i]))
            print '%.3f & %.3f & %.3f' % (np.mean(res2[i]), np.sqrt(np.var(res2[i])), nmad(res2[i]))
            if args.anc_mags:
                print '%.3f & %.3f & %.3f' % (np.mean(res3[i]), np.sqrt(np.var(res3[i])), nmad(res3[i]))
        # delta mag histograms
        fig = plt.figure(figsize=(8,6))
        bins = np.linspace(-1,1,51,endpoint=True)
        plt.hist(res1[:, i], bins=bins, histtype='stepfilled', color='red', alpha=0.3, label='BOSS')
        plt.hist(res2[:, i], bins=bins, histtype='stepfilled', color='blue', alpha=0.3, label='Corrected BOSS')
        if args.anc_mags:
            plt.hist(res3[:, i], bins=bins, histtype='step', color='black', linestyle='dashed', label='Ancillary Reduction')
        plt.grid(True)
        plt.xlim([-1,1])
        plt.xlabel(r'$\Delta{%s}$'%bands[i])
        plt.ylabel('Counts')
        plt.legend()
        #add_stat_legend(res1, prec=3)
        fig.savefig(args.output+'-residuals-%s.pdf' % (bands[i]), bbox_inches='tight')

    color_res1 = res1[:, :-1] - res1[:, 1:]
    color_res2 = res2[:, :-1] - res2[:, 1:]
    if args.anc_mags:
        color_res3 = res3[:, :-1] - res3[:, 1:]

    colors = ['g-r', 'r-i']

    for i in [0, 1]:
        if args.verbose:
            print '  %s:' % colors[i]
            print '%.3f & %.3f & %.3f' % (np.mean(color_res1[i]), np.sqrt(np.var(color_res1[i])), nmad(color_res1[i]))
            print '%.3f & %.3f & %.3f' % (np.mean(color_res2[i]), np.sqrt(np.var(color_res2[i])), nmad(color_res2[i]))
            if args.anc_mags:
                print '%.3f & %.3f & %.3f' % (np.mean(color_res3[i]), np.sqrt(np.var(color_res3[i])), nmad(color_res3[i]))
        # delta mag histograms
        fig = plt.figure(figsize=(8,6))
        xlimit = .5
        bins = np.linspace(-xlimit, xlimit, 41, endpoint=True)
        plt.hist(color_res1[:, i], bins=bins, histtype='stepfilled', color='red', alpha=0.3, label='BOSS')
        plt.hist(color_res2[:, i], bins=bins, histtype='stepfilled', color='blue', alpha=0.3, label='Corrected BOSS')
        if args.anc_mags:
            plt.hist(color_res3[:, i], bins=bins, histtype='step', color='black', linestyle='dashed', label='Ancillary Reduction')
        plt.grid(True)
        plt.xlim([-xlimit, xlimit])
        plt.xlabel(r'$\Delta{(%s)}$' % colors[i])
        plt.ylabel('Counts')
        plt.legend()
        #add_stat_legend(res1, prec=3)
        fig.savefig(args.output+'-residuals-%s.pdf' % (colors[i]), bbox_inches='tight')

        fig = plt.figure(figsize=(8,6))
        plt.scatter(x1[:, 0], color_res1[:, i], facecolor='red', alpha=0.5, label='BOSS')
        plt.scatter(x2[:, 0], color_res2[:, i], facecolor='blue', alpha=0.5, label='Corrected BOSS')
        # if args.anc_mags:
        #     plt.scatter(x3[:, 0], color_res3[:, i], facecolor='white', alpha=0.5, label='Ancillary Reduction')

        contour = density_contour(x1[:, 0], color_res1[:, i], 15, 15, [15.5, 19.5], [-.25, +.4], colors='red', label='BOSS')
        # two_sigma = contour.collections[1]
        # plt.setp(two_sigma, linestyle='dashed')
        contour = density_contour(x2[:, 0], color_res2[:, i], 15, 15, [15.5, 19.5], [-.25, +.4], colors='blue', label='Corrected BOSS')
        # two_sigma = contour.collections[1]
        # plt.setp(two_sigma, linestyle='dashed')
        #contour = density_contour(x3[:, 0], color_res3[:, i], 21, 31, colors='black', label='Ancillary Reduction')
        # two_sigma = contour.collections[1]
        # plt.setp(two_sigma, linestyle='dashed')

        plt.ylim([-xlimit,+xlimit])
        plt.ylabel(r'$\Delta{(%s)}$' % colors[i])
        plt.xlabel(r'$g$')
        plt.legend()
        plt.grid()
        fig.savefig(args.output+'-scatter-%s.pdf' % (colors[i]), bbox_inches='tight')




if __name__ == '__main__':
    main()
