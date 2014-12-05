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
    mask = np.logical_not(np.any(xdata_raw == 0, axis=1)) & np.logical_not(np.any(ydata_raw == 0, axis=1))
    xdata = xdata_raw[mask]
    ydata = ydata_raw[mask]

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
        #axs[i].set_aspect('equal')
        plt.sca(axs[i])
        x = xdata[:,i]
        y = ydata[:,i]
        plt.plot(x, x-y, '+')
        # lower = min(np.min(x), np.min(y))
        # upper = max(np.max(x), np.max(y))
        # diff = upper - lower
        # scale = 0.05
        # pad = scale*diff
        # lim = [lower-pad, upper+pad]
        # plt.ylim(lim)
        # plt.xlim(lim)
        plt.ylim([-1,1])
        plt.grid(True)
        plt.title(xname+' vs. '+yname)
        plt.xlabel(bands[i])
        plt.ylabel(r'$\Delta$%s'%bands[i])
    fig.savefig(args.output+'-gri.png', bbox_inches='tight')

    # delta mag histograms
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot2grid((1,3), (0,0))
    ax2 = plt.subplot2grid((1,3), (0,1))
    ax3 = plt.subplot2grid((1,3), (0,2))
    axs = [ax1, ax2, ax3]
    print 'delta mag (band: mean +/- rms):'
    for i in range(len(axs)):
        plt.sca(axs[i])
        x = xdata[:,i]
        y = ydata[:,i]
        xydiff = x-y
        print '  %s: %.4f +/- %.4f' % (bands[i], np.mean(xydiff), np.sqrt(np.var(xydiff)))
        plt.hist(xydiff, bins=50, lw=0, alpha=.5)
        plt.grid(True)
        plt.xlabel(r'$\Delta$%s'%bands[i])
        plt.xlim([-1,1])
        plt.title(xname+' vs. '+yname)
    fig.savefig(args.output+'-hist.png', bbox_inches='tight')

    # color
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot2grid((1,3), (0,0))
    ax2 = plt.subplot2grid((1,3), (0,1))
    ax3 = plt.subplot2grid((1,3), (0,2))
    axs = [ax1, ax2, ax3]
    print 'delta color (color: mean +/- rms):'
    xaxiscol = 1
    for iax,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
        plt.sca(axs[iax])
        xcolor = xdata[:,i]-xdata[:,j]
        ycolor = ydata[:,i]-ydata[:,j]
        xydiff = xcolor-ycolor
        print '  %s-%s: %.4f +/- %.4f' % (bands[i],bands[j], np.mean(xydiff), np.sqrt(np.var(xydiff)))
        plt.plot(xdata[:,xaxiscol], xydiff, '+')
        plt.grid(True)
        plt.xlabel('%s'%(bands[xaxiscol]))
        plt.ylabel(r'$\Delta(%s-%s)$'%(bands[i],bands[j]))
        plt.title(xname+' vs. '+yname)
    fig.savefig(args.output+'-colors.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
