#!/usr/bin/env python

import argparse
import os
import numpy
from astropy.io import fits

import qusp

def main():

    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save", type=str, default=None,
        help="save plot filename")
    parser.add_argument("--boss-root", type=str, default=None,
        help = "path to root directory containing BOSS data (ex: /data/boss)")
    parser.add_argument("--boss-version", type=str, default="v5_7_0",
        help = "boss pipeline version tag (ex: v5_7_0)")
    parser.add_argument("--target", type=str, default=None,
        help = "target string")
    parser.add_argument("--target-list", type=str, default=None,
        help = "target list filename")
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    fitsPath = os.path.join(boss_root, boss_version)

    if args.target_list:
        targets = qusp.readTargetList(args.target_list)
    elif args.target:
        targets = [qusp.Target.fromString(args.target)]

    if args.save:
        # Force matplotlib to not use any Xwindows backend.
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    if not args.save:
        plt.ion()
        plt.show()

    plateFileName = None
    for target in targets:
        # load the spectrum file
        if plateFileName != 'spPlate-%s-%s.fits' % (target.plate, target.mjd):
            plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
            fullName = os.path.join(fitsPath,str(target.plate),plateFileName)
            print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)

        combined = qusp.readCombinedSpectrum(spPlate, target.fiber)

        fig = plt.figure()
        x = combined.wavelength
        y = combined.flux
        plt.plot(x, y, 'b+')
        plt.xlim([x[0], x[-1]])
        plt.title(r'%s' % str(target))
        plt.xlabel(r'Wavelength $(\AA)$')
        plt.ylabel(r'Flux $(10^{-17} erg/cm^2/s/\AA)$')

        if args.save:
            fig.savefig(args.save, bbox_inches='tight')
        else:
            plt.draw()
            raw_input('hit [ENTER]...')


if __name__ == '__main__':
    main()