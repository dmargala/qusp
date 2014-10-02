#!/usr/bin/env python
"""
Calculates SN ratio of BOSS spectra
"""

import os
import argparse

import numpy as np

from astropy.io import fits

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

    paths = qusp.Paths(**qusp.Paths.fromArgs(args))

    # read target list
    targets = qusp.target.loadTargetData(args.input)
    ntargets = len(targets)

    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    # loop over targets
    plateFileName = None
    for targetIndex in range(ntargets):
        if args.ntargets > 0 and targetIndex > args.ntargets:
            break
        target = targets[targetIndex]

        plate,mjd,fiber = target['target'].split('-')

        # load the spectrum file
        if plateFileName != 'spPlate-%s-%s.fits' % (plate, mjd):
            plateFileName = 'spPlate-%s-%s.fits' % (plate, mjd)
            fullName = os.path.join(paths.boss_path,plate,plateFileName)
            if args.verbose:
                print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)

        # read this target's combined spectrum
        combined = qusp.readCombinedSpectrum(spPlate, int(fiber))

        mediansn = combined.getMedianSignalToNoise(combined.wavelength[0],combined.wavelength[-1])

        target['mediansn'] = mediansn
        print target



if __name__ == '__main__':
    main()