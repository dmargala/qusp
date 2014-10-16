#!/usr/bin/env python
"""
Stacks boss spectra.
"""

import argparse
import numpy as np
import h5py

import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, default=None,
        help="target list")
    parser.add_argument("-n", "--ntargets", type=int, default=0,
        help="number of targets to use, 0 for all")
    parser.add_argument("-o", "--output", type=str, default=None,
        help="output file name")
    parser.add_argument("--verbose", action="store_true",
        help="more verbose output")
    parser.add_argument("--select-plate", type=int, default=None,
        help="select targets from specified plate")
    parser.add_argument("--normmin", type=float, default=5000,
        help="min norm wavelength")
    parser.add_argument("--normmax", type=float, default=5010,
        help="max norm wavelength")
    qusp.Paths.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    # read target list
    targets = qusp.target.load_target_list(args.input)

    if args.select_plate:
        plate_targets = []
        for target in targets:
            if target['plate'] == args.select_plate:
                plate_targets.append(target)
        targets = plate_targets

    # trim target list if requested
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)
    targets = targets[:ntargets]

    # initialize stack arrays
    npixels_fiducial = 4800
    wavelength = qusp.wavelength.get_fiducial_wavelength(np.arange(npixels_fiducial))

    flux_wsum = np.zeros(npixels_fiducial)
    weight_sum = np.zeros_like(flux_wsum)

    # loop over targets
    target_plate_generator = qusp.target.read_target_plates(paths.boss_path, targets, verbose=args.verbose)
    for target, spplate in target_plate_generator:
        # read this target's combined spectrum
        combined = qusp.read_combined_spectrum(spplate, target)

        norm = combined.mean_flux(args.normmin, args.normmax)

        if norm <= 0:
            print 'yikes norm <= 0'
            continue

        offset = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(combined.wavelength[0]))

        indices = slice(offset, offset+combined.npixels)

        flux_wsum[indices] += combined.ivar*combined.flux/norm
        weight_sum[indices] += combined.ivar

    flux_wmean = np.empty_like(flux_wsum)
    nonzero_weights = np.nonzero(weight_sum)
    flux_wmean[nonzero_weights] = flux_wsum[nonzero_weights]/weight_sum[nonzero_weights]

    if args.output:
        outfilename = args.output+'.hdf5'
        if args.verbose:
            print 'Saving stack to file: %s' % outfilename
        # save target list with sn column
        outfile = h5py.File(outfilename, 'w')

        outfile.create_dataset('flux_wmean', data=flux_wmean)
        outfile.create_dataset('weight_sum', data=weight_sum)
        outfile.create_dataset('wavelength', data=wavelength)
        outfile.attrs['ntargets'] = ntargets


if __name__ == '__main__':
    main()
