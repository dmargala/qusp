#!/usr/bin/env python
"""
Calculates SN ratio of BOSS spectra
"""

import argparse
import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--input", type=str, default=None,
        help = "target list")
    parser.add_argument("-n","--ntargets", type=int, default=0,
        help = "number of targets to use, 0 for all")
    parser.add_argument("-o","--output", type=str, default="sn.txt",
        help = "output file name")
    parser.add_argument("--verbose", action="store_true",
        help = "more verbose output")
    qusp.Paths.addArgs(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.fromArgs(args))
    # read target list
    targets = qusp.target.loadTargetData(args.input)

    # trim target list if requested
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)
    targets = targets[:ntargets]

    # loop over targets
    for target, spPlate in qusp.target.readTargetPlates(paths.boss_path,targets,verbose=args.verbose):
        plate, mjd, fiber = target['target'].split('-')
        # read this target's combined spectrum
        combined = qusp.readCombinedSpectrum(spPlate, int(fiber))
        # calculate median sn
        mediansn = combined.getMedianSignalToNoise(combined.wavelength[0],combined.wavelength[-1])
        target['mediansn'] = mediansn
    # save target list with sn column
    qusp.target.saveTargetData(args.output, targets, ['mediansn'])


if __name__ == '__main__':
    main()