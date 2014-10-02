#!/usr/bin/env python

import argparse

import numpy
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import h5py

def plothist2d(data, savename, xdata, ydata, label, **kwargs):

    # x,y grid characterizing stack pixels' corners
    # the fiducial wavelength solution specifies bin centers, so subtract half a pixel
    x = xdata.value
    y = ydata.value
    X,Y = numpy.meshgrid(x,y)

    # 2D histogram
    fig = plt.figure(figsize=(24,10))
    ax1 = fig.add_subplot(111)

    # clip color manually, reduce colorbar pad from 0.05 -> 0.01
    mappable = ax1.pcolormesh(X, Y, numpy.transpose(data), **kwargs)
    cbar = fig.colorbar(mappable, pad=.01)
    cbar.set_label(label)

    # Label standard axes
    ax1.set_xlabel(xdata.attrs['label'])
    ax1.set_xlim(x[0],x[-1])
    ax1.set_ylabel(ydata.attrs['label'])
    ax1.set_ylim(y[0],y[-1])

    # Add restframe labels to upper x axis 
    if 'x2label' in xdata.attrs.keys():
        ax2 = ax1.twiny()
        ax2.set_xlabel(xdata.attrs['x2label'], color='r')
        ax2.set_xlim([xdata.attrs['x2min'],xdata.attrs['x2max']])
        for tl in ax2.get_xticklabels():
            tl.set_color('r')

    fig.savefig(savename, bbox_inches='tight')

def main():

    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--input", type=str, default="stack.hdf5",
        help="stacked spectrum npy file")
    parser.add_argument("--save", type=str, default="stack",
        help="save plot filename")
    parser.add_argument("--vmax", type=float, default=0,
        help = "upper flux limit")

    args = parser.parse_args()

    infile = h5py.File(args.input,'r')

    hists2d = infile['hists2d']
    xdata = hists2d['xbinedges']
    ydata = hists2d['ybinedges']

    vmax = args.vmax

    for name in ('fluxmean','wfluxmean','counts','weights','wfluxvar','sn','pullvar','pullmean'):

        data = hists2d[name].value
        label = hists2d[name].attrs['label']

        if name == 'pullvar':
            mean = hists2d['pullmean'].value
            data = numpy.sqrt(data - mean**2)
            name = 'pullrms'
            label = 'Pull RMS'

        vmax = numpy.percentile(data[data>0],99)
        vmin = 0 if name != 'pullmean' else -vmax

        cmap = plt.get_cmap('bwr' if name == 'pullmean' else 'Blues')

        savename = '%s_%s.png' % (args.save, name)

        plothist2d(data, savename, xdata, ydata, label, vmin=vmin, vmax=vmax, cmap=cmap)


if __name__ == '__main__':
    main()