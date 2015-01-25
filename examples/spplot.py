#!/usr/bin/env python

import argparse
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py

import qusp

from scipy.signal import savgol_filter

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="more verbose output")
    parser.add_argument("--wave-min", type=float, default=3600,
        help="wavelength min")
    parser.add_argument("--wave-max", type=float, default=10500,
        help="wavelength max")
    parser.add_argument("--output", type=str, default=None,
        help="output filename")
    parser.add_argument("--tpcorr", type=str, default=None,
        help="throughput correction filename")
    parser.add_argument("--nsmooth", type=int, default=5,
        help="savgol filter length in pixels")
    qusp.paths.Paths.add_args(parser)
    qusp.target.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))

    # wavelength range limits
    wave_min = qusp.wavelength.Wavelength(args.wave_min)
    wave_max = qusp.wavelength.Wavelength(args.wave_max)

    # read target
    if args.targets:
        target_list = qusp.target.load_target_list_from_args(args)
    else:
        raise RuntimeError('Invalid target specification.')

    fig = plt.figure(figsize=(16, 4))

    for target in target_list:
        # load target's spectrum
        combined = qusp.target.get_combined_spectrum(target, paths)

        x = combined.wavelength
        y = combined.flux.values

        plt.plot(savgol_filter(x, args.nsmooth, 2), savgol_filter(y, args.nsmooth, 2), 
            lw=.5, label=target.to_string())

    plt.legend()

    plt.xlim([wave_min, wave_max])
    ymin = 0
    ymax = 15
    plt.ylim([ymin, ymax])

    plt.ylabel(r'Flux $(10^{-17} erg/cm^2/s/\AA)$')
    plt.xlabel(r'Observed Wavlength $(\AA)$')

    plt.grid()

    filename = args.output
    if filename is None:
        filename = args.targets[:-3]+'png'
    fig.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':
    main()
