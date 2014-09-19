#!/usr/bin/env python
import argparse
import os

import numpy as np
import h5py
from astropy.io import fits

import matplotlib.pyplot as plt

import bosslya

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
    parser.add_argument("--random", action="store_true",
        help="use a random selection of input targets")
    parser.add_argument("--seed", type=int, default=42,
        help="rng seed")

    bosslya.ContinuumFitter.addArgs(parser)
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    fitsPath = os.path.join(boss_root, boss_version)

    # read target list
    targets = bosslya.readTargetList(args.input,[('ra',float),('dec',float),('z',float),('thingid',int),('sn',float)])
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)

    # use the first n targets or a random sample
    if args.random:
        np.random.seed(args.seed)
        targets = [targets[i] for i in np.random.randint(len(targets), size=ntargets)]
    else:
        targets = targets[:ntargets]

    # we want to open the spPlate files in plate-mjd order
    targets = sorted(targets)

    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    # Initialize fitter 
    fitter = bosslya.ContinuumFitter(args.obsmin, args.obsmax, 
        args.restmin, args.restmax, args.nrestbins, nuWave=args.nuwave,
        alphaMin=args.alphamin, alphaMax=args.alphamax,
        beta=args.beta, verbose=args.verbose)

    if args.verbose:
        print 'Fit model initialized with %d model params.\n' % fitter.nModelPixels
        print '... adding observations to fit ...\n'

    # Add observations to fitter
    plateFileName = None
    fitTargets = []
    npixels = []
    for targetIndex, target in enumerate(targets):
        # load the spectrum file
        if plateFileName != 'spPlate-%s-%s.fits' % (target.plate, target.mjd):
            if plateFileName:
                spPlate.close()
            plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
            fullName = os.path.join(fitsPath,str(target.plate),plateFileName)
            # if args.verbose:
            #    print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)

        # read this target's combined spectrum
        combined = bosslya.readCombinedSpectrum(spPlate, target.fiber)
        wavelength = combined.wavelength
        ivar = combined.ivar
        flux = combined.flux

        # Add this observation to our fitter
        nPixelsAdded = fitter.addObservation(target, flux, wavelength, ivar, 
            unweighted=args.unweighted)
        if nPixelsAdded > 0:
            fitTargets.append(target)
            npixels.append(nPixelsAdded)

    if args.verbose:
        print ''

    # Add constraints
    if args.restnormmax > args.restnormmin:
        fitter.addRestConstraint(0, args.restnormmin, args.restnormmax, args.restnormweight)
    if args.obsnormmax > args.obsnormmin:
        fitter.addObsConstraint(0, args.obsnormmin, args.obsnormmax, args.obsnormweight)
    if args.nuweight > 0:
        fitter.addNuConstraint(args.nuweight)

    if args.verbose:
        print ''

    # run the fitter
    results = fitter.fit(atol=args.atol, btol=args.btol, max_iter=args.max_iter, sklearn=args.sklearn)
    chisq = fitter.getChiSq()

    if args.verbose:
        print 'chisq (nModelParams,nConstraints): %.2g (%d,%d)' % (chisq, fitter.model.shape[1], fitter.nconstraints)
        print 'reduced chisq: %.2g' % (chisq/(fitter.model.shape[1]-fitter.nconstraints))

    # Save HDF5 file with results
    outfile = h5py.File(args.output,'w')

    fitter.save(outfile, args)

    outfile.create_dataset('npixels', data=npixels)
    outfile.create_dataset('targets', data=[str(target) for target in fitTargets])
    outfile.create_dataset('redshifts', data=[target.z for target in fitTargets])
    outfile.create_dataset('sn', data=[target.sn for target in fitTargets])

    outfile.close()

if __name__ == '__main__':
    main()