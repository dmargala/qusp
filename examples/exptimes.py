#!/usr/bin/env python
import argparse
import os

import numpy as np
import h5py
from astropy.io import fits

import matplotlib.pyplot as plt

import qusp

import desimodel.simulate

from astropy import units as u

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-o","--output", type=str, default=None,
        help="hdf5 output filename")
    ## BOSS data
    parser.add_argument("--boss-root", type=str, default=None,
        help="path to root directory containing BOSS data (ex: /data/boss)")
    parser.add_argument("--boss-version", type=str, default="v5_7_0",
        help="boss pipeline version tag")
    ## targets to fit
    parser.add_argument("-i","--input", type=str, default=None,
        help="target list")
    parser.add_argument("-n","--ntargets", type=int, default=0,
        help="number of targets to use, 0 for all")
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    fitsPath = os.path.join(boss_root, boss_version)

    # read target list
    targets = qusp.target.loadTargetData(args.input)
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)

    # we want to open the spPlate files in plate-mjd order
    targets = sorted(targets[:ntargets])

    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    # Add observations to fitter
    currentlyOpened = None
    for targetIndex, target in enumerate(targets):
        plate, mjd, fiber = target['target'].split('-')
        plateFileName = 'spPlate-%s-%s.fits' % (plate, mjd)
        # load the spectrum file
        if plateFileName != currentlyOpened:
            if currentlyOpened is not None:
                spPlate.close()
            fullName = os.path.join(fitsPath,plate,plateFileName)
            # if args.verbose:
            #    print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)
            currentlyOpened = plateFileName

            taibeg = spPlate[0].header['TAI-BEG']
            taiend = spPlate[0].header['TAI-END']
            nexp = spPlate[0].header['NEXP']/4
            ra = spPlate[0].header['RA']
            dec = spPlate[0].header['RA']
            print taibeg, taiend, taiend-taibeg, nexp, ra

            plateDiameter = (3.0*u.deg).to(u.rad)

            fig = plt.figure(figsize=(8,8))
            ax = plt.subplot(111,polar=True)
            #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
            bars = ax.bar(ra, 2.5, width=plateDiameter.value, bottom=0.0, alpha=.5)
            plt.tick_params(labelleft='off')
            plt.ylim([0,4])
            # Use custom colors and opacity
            # for r, bar in zip(radii, bars):
            #     bar.set_facecolor(plt.cm.jet(r / 10.))
            #     bar.set_alpha(0.5)

            plt.show()


if __name__ == '__main__':
    main()