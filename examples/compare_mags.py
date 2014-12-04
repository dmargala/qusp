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
    parser.add_argument("-i1", "--input1", type=str, default=None,
        help="required input file")
    parser.add_argument("-i2", "--input2", type=str, default=None,
        help="required input file")
    args = parser.parse_args()

    if args.verbose:
        print 'Reading files: %s, %s' % (args.input1, args.input2)

    data1 = np.loadtxt(args.input1)
    data2 = np.loadtxt(args.input2)

    bands = 'gri'

    mask = np.logical_not(np.any(data1 == 0, axis=1)) & np.logical_not(np.any(data2 == 0, axis=1))

    xdata = data1[mask]
    ydata = data2[mask]

    if args.verbose:
        print 'Creating summary plot...'

    # save results summary plot
    fig = plt.figure(figsize=(12,4))

    ax1 = plt.subplot2grid((1,3), (0,0))
    ax2 = plt.subplot2grid((1,3), (0,1))
    ax3 = plt.subplot2grid((1,3), (0,2))

    axs = [ax1, ax2, ax3]

    for i in range(len(axs)):
        axs[i].set_aspect('equal')
        plt.sca(axs[i])
        x = xdata[:,i]
        y = ydata[:,i]
        xydiff = x-y
        print '%.4f %.4f' % (np.mean(xydiff), np.sqrt(np.var(xydiff)))
        lower = min(np.min(x), np.min(y))
        upper = max(np.max(x), np.max(y))
        diff = upper - lower
        scale = 0.05
        pad = scale*diff
        lim = [lower-pad, upper+pad]
        plt.plot(x, y, '+')
        plt.ylim(lim)
        plt.xlim(lim)
        plt.grid(True)
        plt.title(bands[i])

        plt.xlabel(args.input1)
        plt.ylabel(args.input2)

    fig.savefig(args.output+'-gri.png', bbox_inches='tight')

    fig = plt.figure(figsize=(6,6))

    plt.plot(xdata[:,0]-xdata[:,2], ydata[:,0]-ydata[:,2], '+')
    plt.grid(True)
    plt.title('g-i')

    plt.xlabel(args.input1)
    plt.ylabel(args.input2)

    fig.savefig(args.output+'-gminusi.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
