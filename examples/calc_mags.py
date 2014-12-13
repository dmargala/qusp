#!/usr/bin/env python
"""
"""
import numpy as np

import argparse
import qusp
import glob

import scipy.interpolate

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output", type=str, default=None,
        help="output file base name")
    parser.add_argument("--tpcorr", type=str, default=None,
        help="throughput correction filename")
    qusp.target.add_args(parser)
    qusp.Paths.add_args(parser)
    args = parser.parse_args()

    if args.tpcorr:
        import h5py
        tpcorr = h5py.File(args.tpcorr)

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))

    # read target list
    targets = qusp.target.load_target_list_from_args(args,
        fields=[(band, float, i+1) for i,band in enumerate('ugriz')])

    # loop over targets
    mags = []
    for target in targets:
        if args.tpcorr:
            combined = qusp.target.get_corrected_spectrum(target, tpcorr, paths=paths)
        else:
            combined = qusp.target.get_combined_spectrum(target, paths=paths)
        ab_mags = combined.flux.get_ab_magnitudes()
        save_mags = []
        for band in 'gri':
            if ab_mags[band] is None:
                print target, band
                save_mags.append(0)
            else:
                save_mags.append(ab_mags[band])
        mags.append(save_mags)
    np.savetxt(args.output, mags)

if __name__ == '__main__':
    main()
