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
    # z evolution options
    parser.add_argument("--alpha", action="store_true",
        help="add z evolution term")
    parser.add_argument("--beta", type=float, default=3.92,
        help="optical depth power law parameter")
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
    targets = sorted(targets[:ntargets])
    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    # prepare model
    params = []
    params.append({'name':'T','type':'obs'})
    params.append({'name':'C','type':'rest'})

    if args.alpha:
        if args.verbose:
            print 'Adding z evo param with beta = %.2f' % args.beta
        def alphaCoef(obs, rest):
            return -np.power(obs/rest, args.beta)
        params.append({'name':'alpha','type':'rest','coef':alphaCoef})

    params.append({'name':'A','type':'target'})

    # Initialize fitter 
    fitter = bosslya.ContinuumFitter(params, args.obsmin, args.obsmax, 
        args.restmin, args.restmax, args.nrestbins, verbose=args.verbose)

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
            plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
            fullName = os.path.join(fitsPath,str(target.plate),plateFileName)
            #if args.verbose:
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

    if args.restnorm > 0:
        fitter.addConstraint('C', 0, args.restnorm, args.drestnorm, args.restnormweight)
    if args.obsnorm > 0:
        fitter.addConstraint('T', 0, args.obsnorm, args.dobsnorm, args.obsnormweight)

    # run the fitter
    results = fitter.fit(atol=args.atol, btol=args.btol, max_iter=args.max_iter, sklearn=args.sklearn)
    chisq = fitter.getChiSq()

    if args.verbose:
        print 'chisq (nModelParams,nConstraints): %.2f (%d,%d)' % (chisq, fitter.model.shape[1], fitter.nconstraints)
        print 'reduced chisq: %.2f' % (chisq/(fitter.model.shape[1]-fitter.nconstraints))

    # transform fit results
    obsModelValues = np.exp(results['T'])
    restModelValues = np.exp(results['C'])
    targetModelValues = np.exp(results['A'])

    if args.alpha:
        alphaModelValues = results['alpha']

    # Save HDF5 file with results
    outfile = h5py.File(args.output,'w')

    outfile.create_dataset('npixels', data=npixels)
    outfile.create_dataset('model_data', data=fitter.model.data)
    outfile.create_dataset('model_indices', data=fitter.model.indices)
    outfile.create_dataset('model_indptr', data=fitter.model.indptr)
    outfile.create_dataset('model_shape', data=fitter.model.shape)

    outfile.create_dataset('obsWaveCenters', data=fitter.obsWaveCenters)
    outfile.create_dataset('restWaveCenters', data=fitter.restWaveCenters)

    outfile.create_dataset('targets', data=[str(target) for target in fitTargets])
    outfile.create_dataset('redshifts', data=[target.z for target in fitTargets])
    outfile.create_dataset('sn', data=[target.sn for target in fitTargets])

    outfile.create_dataset('chisq', data=[fitter.getObservationChiSq(i) for i in range(len(fitTargets))])

    outfile.create_dataset('T', data=obsModelValues)
    outfile.create_dataset('C', data=restModelValues)
    outfile.create_dataset('A', data=targetModelValues)

    if args.alpha:
        outfile.create_dataset('alpha', data=alphaModelValues)

    outfile.close()

if __name__ == '__main__':
    main()