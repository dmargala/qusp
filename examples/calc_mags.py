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

    tpcorr_map = {}
    if args.tpcorr:
        tpcorr_filenames = glob.glob(args.tpcorr)

        for tpcorr_filename in tpcorr_filenames:
            plate,mjd = tpcorr_filename.split('.')[0].split('-')[1:]

            # print plate, mjd, 

            # The input data is text file where each line coresponds to 
            # a target's throughput correction vector
            data = np.loadtxt(tpcorr_filename)
            nentries, ntokens = data.shape

            # the first 3 columns are fiberid, xfocal, and yfocal positions of the target
            nidtokens = 3
            # the rest are the tabulated throughput correction values
            npoints = ntokens - nidtokens

            # the throughput correction vectors span the range 3500A to 10500A
            xvalues = np.linspace(3500, 10500, npoints, endpoint=True)

            for row in data:
                fiberid = str(int(row[0]))
                # print fiberid,
                target = '-'.join([plate,mjd,fiberid])
                tpcorr_values = row[nidtokens:]
                tpcorr_map[target] = scipy.interpolate.interp1d(xvalues, tpcorr_values,
                    kind='linear', copy=False)
            # print '\n',

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))

    # read target list
    targets = qusp.target.load_target_list_from_args(args,
        fields=[(band, float, i+1) for i,band in enumerate('ugriz')])

    # loop over targets
    mags = []
    for target, combined in qusp.target.get_combined_spectra(targets, boss_path=paths.boss_path):
        if target.to_string() in tpcorr_map.keys():
            corrected = combined.create_corrected(tpcorr_map[target.to_string()])
            ab_mags = corrected.flux.get_ab_magnitudes()
        else:
            if args.tpcorr:
                print 'No tpcorr for target: %s' % target.to_string()
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
