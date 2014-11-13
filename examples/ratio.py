#!/usr/bin/env python
"""
Stacks boss spectra.
"""

import argparse
import numpy as np
import h5py

import qusp

import itertools

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
    parser.add_argument("--compare-version", type=str, default=None,
        help="boss version to compare")
    qusp.Paths.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    alt_paths = qusp.Paths(boss_version=args.compare_version)

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

    # open output file
    outfilename = args.output+'.hdf5'
    outfile = h5py.File(outfilename, 'w')

    # loop over targets
    target_combined_generator = qusp.target.get_combined_spectra(targets, boss_path=paths.boss_path, verbose=args.verbose)
    alt_target_combined_generator = qusp.target.get_combined_spectra(targets, boss_path=alt_paths.boss_path, verbose=args.verbose)

    for (target, combined), (alt_target, alt_combined) in itertools.izip(target_combined_generator, alt_target_combined_generator):
        # make sure we're comparing the same target
        assert target.to_string() == alt_target.to_string()

        offset = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(combined.wavelength[0]))
        alt_offset = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(alt_combined.wavelength[0]))

        first_pixel = offset - alt_offset

        if first_pixel > 0:
            npixels = min(combined.npixels, alt_combined.npixels-first_pixel)
            ratio = alt_combined.flux[first_pixel:first_pixel+npixels]/combined.flux[:npixels]
            wavelength = combined.wavelength[:npixels]
        else:
            first_pixel = -first_pixel
            npixels = min(combined.npixels-first_pixel, alt_combined.npixels)
            ratio = alt_combined.flux[:npixels]/combined.flux[first_pixel:first_pixel+npixels]
            wavelength = alt_combined.wavelength[:npixels]

        grp = outfile.create_group(target.to_string())
        grp.create_dataset('ratio', data=ratio)
        grp.create_dataset('wavelength', data=wavelength)

    outfile.create_dataset('targets', data=[target.to_string() for target in targets])


if __name__ == '__main__':
    main()
