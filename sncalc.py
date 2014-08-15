#!/usr/bin/env python
"""
Calculates SN ratio of BOSS spectra
"""

import os
import argparse

import numpy as np

from astropy.io import fits

import bosslya

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--input", type=str, default=None,
        help = "target list")
    parser.add_argument("-n","--ntargets", type=int, default=0,
        help = "number of targets to use, 0 for all")
    parser.add_argument("--boss-root", type=str, default=None,
        help = "path to root directory containing BOSS data (ex: /data/boss)")
    parser.add_argument("--boss-version", type=str, default=None,
        help = "boss pipeline version tag (ex: v5_7_0)")
    parser.add_argument("-o","--output", type=str, default="sn.txt",
        help = "output file name")
    parser.add_argument("--verbose", action="store_true",
        help = "more verbose output")
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    fitsPath = os.path.join(boss_root, boss_version)

    # read target list
    targets = bosslya.readTargetList(args.input,[('ra',float),('dec',float),('z',float),('thingid',int)])
    ntargets = len(targets)

    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    skipcounter = 0
    sncounter = 0

    # loop over targets
    plateFileName = None
    for targetIndex in range(ntargets):
        if args.ntargets > 0 and targetIndex > args.ntargets:
            break
        target = targets[targetIndex]

        # load the spectrum file
        if plateFileName != 'spPlate-%s-%s.fits' % (target.plate, target.mjd):
            plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
            fullName = os.path.join(fitsPath,str(target.plate),plateFileName)
            if args.verbose:
                print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)

        # read this target's combined spectrum
        combined = bosslya.readCombinedSpectrum(spPlate, target.fiber)

        mediansn = combined.getMedianSignalToNoise(combined.wavelength[0],combined.wavelength[-1])

        print str(target),
        for attr in target.attrs():
            print attr,
        print mediansn



if __name__ == '__main__':
    main()