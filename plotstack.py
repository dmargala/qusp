#!/usr/bin/env python

import argparse

import numpy
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import h5py

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
    zmin = hists2d.attrs['zmin']
    zmax = hists2d.attrs['zmax']

    for name in ('fluxmean','wfluxmean','wfluxvar','counts','weights','sn','pullvar','pullmean'):

        data = hists2d[name].value
        label = hists2d[name].attrs['label']

        nzbins = data.shape[1]
        npixels = data.shape[0]

        # x,y grid chracterizing stack pixels' corners
        # the fiducial wavelength solution specifies bin centers, so subtract half a pixel
        x = 3500.26*numpy.power(10, 1e-4*(numpy.arange(0, npixels+1)-0.5))
        y = numpy.linspace(zmin, zmax, nzbins+1, endpoint=True)
        X,Y = numpy.meshgrid(x,y)

        # 2D histogram
        fig = plt.figure(figsize=(24,10))
        ax1 = fig.add_subplot(111)

        vmax = args.vmax if args.vmax > 0 else numpy.percentile(data,99)
        vmin = 0 if name != 'pullmean' else -vmax

        if name == 'pullvar':
            vmax = 5
            mean = hists2d['pullmean'].value
            data = numpy.sqrt(data - mean**2)
            name = 'pullrms'
            label = 'Pull RMS'

        cmap = plt.get_cmap('Blues')
        if name == 'pullmean':
            cmap = plt.get_cmap('bwr')

        # clip color manually, reduce colorbar pad from 0.05 -> 0.01
        mappable = ax1.pcolormesh(X, Y, numpy.transpose(data), vmin=vmin, vmax=vmax, cmap=cmap)
        cbar = fig.colorbar(mappable, pad=.01)
        cbar.set_label(label)

        # Label standard axes
        ax1.set_ylabel('Redshift (z)')
        ax1.set_xlabel('Observed (z = 0) Wavelength ($\AA$)')
        ax1.set_ylim(zmin,zmax)
        ax1.set_xlim(x[0],x[-1])

        # Add restframe labels to upper x axis 
        ax2 = ax1.twiny()
        ax2.set_xlabel('Restframe (z = %.1f) Wavelength ($\AA$)' % zmax, color='r')
        ax2.set_xlim([x[0]/(1+zmax),x[-1]/(1+zmax)])
        for tl in ax2.get_xticklabels():
            tl.set_color('r')

        fig.savefig('%s_%s.png' % (args.save, name), bbox_inches='tight')

    fig = plt.figure()
    forest = infile['forest'].value
    nonforest = infile['nonforest'].value
    plt.plot(forest,'b+')
    plt.plot(nonforest,'r+')

    ymax = 1.1*max(numpy.percentile(forest,99.5),numpy.percentile(nonforest,99.5))

    plt.ylim([-ymax,ymax])
    plt.xlim([0,len(forest)])

    fig.savefig('%s_%s.png' % (args.save, '2.4'), bbox_inches='tight')


if __name__ == '__main__':
    main()