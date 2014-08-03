#!/usr/bin/env python

import argparse

import numpy
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():

    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--input", type=str, default="stack.npy",
        help="stacked spectrum npy file")
    parser.add_argument("--save", type=str, default="stack.pdf",
        help="save plot filename")
    parser.add_argument("--zmin", type=float, default=2.1,
        help = "minimum quasar redshift to include")
    parser.add_argument("--zmax", type=float, default=3,
        help = "maximum quasar redshift to include")
    parser.add_argument("--vmax", type=float, default=10,
        help = "upper flux limit")

    args = parser.parse_args()

    with open(args.input) as infile:
        wstack = numpy.load(infile)

    # 2D histogram
    fig = plt.figure(figsize=(24,5))
    ax1 = fig.add_subplot(111)

    zmin = args.zmin
    zmax = args.zmax

    nzbins = wstack.shape[1]
    npixels = wstack.shape[0]

    # x,y grid chracterizing stack pixels' corners
    # the fiducial wavelength solution specifies bin centers, so subtract half a pixel
    x = 3500.26*numpy.power(10, 1e-4*(numpy.arange(0, npixels+1)-0.5))
    y = numpy.linspace(zmin, zmax, nzbins+1, endpoint=True)
    X,Y = numpy.meshgrid(x,y)

    # clip color manually, reduce colorbar pad from 0.05 -> 0.01
    mappable = ax1.pcolormesh(X, Y, wstack.T, vmin=0, vmax=args.vmax)
    cbar = fig.colorbar(mappable, pad=.01)
    cbar.set_label('Mean Flux ($10^{-17} erg/cm^2/s/\AA$)')

    # Label standard axes
    ax1.set_ylabel('Redshift (z)')
    ax1.set_xlabel('Observed (z = 0) Wavelength ($\AA$)')
    ax1.set_ylim(zmin,zmax)
    ax1.set_xlim(x[0],x[-1])

    # Add restframe labels to upper x axis 
    ax2 = ax1.twiny()
    ax2.set_xlabel('Restframe (z = %d) Wavelength ($\AA$)' % zmax, color='r')
    ax2.set_xlim([x[0]/(1+zmax),x[-1]/(1+zmax)])
    for tl in ax2.get_xticklabels():
        tl.set_color('r')

    if args.save:
        fig.savefig(args.save, bbox_inches='tight')

if __name__ == '__main__':
    main()