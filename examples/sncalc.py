#!/usr/bin/env python
"""
Calculates SN ratio of BOSS spectra
"""

import argparse
import qusp

def main():
    """
    Main program
    """
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
    qusp.Paths.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    # read target list
    targets = qusp.target.load_target_list(args.input)

    # trim target list if requested
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)
    targets = targets[:ntargets]

    # loop over targets
    target_plate_generator = qusp.target.read_target_plates(
        paths.boss_path, targets, verbose=args.verbose)
    for target, spplate in target_plate_generator:
        # read this target's combined spectrum
        combined = qusp.read_combined_spectrum(spplate, target)
        # calculate median sn
        mediansn = combined.median_signal_to_noise(
            combined.wavelength[0], combined.wavelength[-1])
        target['mediansn'] = mediansn
    if args.output:
        # save target list with sn column
        qusp.target.save_target_list(args.output, targets, 'mediansn')
    else:
        for target in targets:
            print '%s %.2f' % (target.to_string(), target['mediansn'])


if __name__ == '__main__':
    main()
