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
    ####### fit options #######
    parser.add_argument("--unweighted", action="store_true",
        help="perform unweighted least squares fit")
    # z evolution options
    parser.add_argument("--alpha", action="store_true",
        help="add z evolution term")
    parser.add_argument("--beta", type=float, default=3.92,
        help="optical depth power law parameter")
    # transmission model wavelength grid options
    parser.add_argument("--transmin", type=float, default=3600,
        help="transmission model wavelength minimum")
    parser.add_argument("--transmax", type=float, default=10000,
        help="transmission model wavelength maximum")
    # continuum model wavelength grid options
    parser.add_argument("--restmin", type=float, default=850,
        help="rest wavelength minimum")
    parser.add_argument("--restmax", type=float, default=2850,
        help="rest wavelength maximum")
    parser.add_argument("--nrestbins", type=int, default=500,
        help="number of restframe bins")
    ## continuum model constraint
    parser.add_argument("--restnorm", type=float, default=1280,
        help="restframe wavelength to normalize at")
    parser.add_argument("--drestnorm", type=float, default=10,
        help="restframe window size +/- on each side of restnorm wavelength")
    parser.add_argument("--restnormweight", type=float, default=1e4,
        help="norm constraint weight")
    ## transmission model constraint
    parser.add_argument("--obsnorm", type=float, default=5000,
        help="obsframe wavelength to normalize at")
    parser.add_argument("--dobsnorm", type=float, default=10,
        help="obsframe window size +/- on each side of obsnorm wavelength")
    parser.add_argument("--obsnormweight", type=float, default=1e4,
        help="norm constraint weight")
    bosslya.ContinuumFitter.addArgs(parser)
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    fitsPath = os.path.join(boss_root, boss_version)

    # read target list
    targets = bosslya.readTargetList(args.input,[('ra',float),('dec',float),('z',float),('thingid',int)])
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)
    targets = sorted(targets[:ntargets])
    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    # initialize binning arrays
    nobsbins = 4800
    obsWaveCenters = bosslya.getFiducialWavelength(np.arange(nobsbins))

    transminindex = np.argmax(obsWaveCenters > args.transmin)
    transmaxindex = np.argmax(obsWaveCenters > args.transmax)
    transWaveCenters = bosslya.getFiducialWavelength(np.arange(transminindex,transmaxindex+1))
    ntransbins = len(transWaveCenters)
    print transWaveCenters
    if args.verbose:
        print 'Observed frame bin centers span [%.2f,%.2f] with %d bins.' % (
            transWaveCenters[0],transWaveCenters[-1],ntransbins)

    restmin = args.restmin
    restmax = args.restmax
    nrestbins = args.nrestbins
    drest = float(restmax-restmin)/nrestbins
    restWaveCenters = np.linspace(restmin,restmax,nrestbins,endpoint=False) + drest/2
    if args.verbose:
        print 'Rest frame bin centers span [%.2f,%.2f] with %d bins.' % (
            restWaveCenters[0],restWaveCenters[-1],nrestbins)

    # Initialize model
    params = []
    params.append({'name':'T','type':'obs'})
    params.append({'name':'C','type':'rest'})
    params.append({'name':'A','type':'target'})

    if args.alpha:
        if args.verbose:
            print 'Adding z evo param with beta = %.2f' % args.beta
        def alphaCoef(obs, rest):
            return -np.power(obs/rest, args.beta)
        params.append({'name':'alpha','type':'rest','coef':alphaCoef})

    # Initialize fitter 
    fitter = bosslya.ContinuumFitter(params, transWaveCenters, restWaveCenters)

    if args.verbose:
        print 'Fit model initialized with %d model params.' % fitter.nModelPixels

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

        # this spectrum's wavelength axis pixel offset
        offset = bosslya.getFiducialPixelIndexOffset(np.log10(wavelength[0]))

        obsindices = np.arange(offset,offset+combined.nPixels)
        obswave = obsWaveCenters[obsindices]
        restwave = obswave/(1+target.z)
        restindices = ((restwave - restmin)/(restmax - restmin)*nrestbins).astype(int)

        # trim ranges to valid data
        validbins = np.all((
            obsindices >= transminindex, 
            obsindices  < transminindex+ntransbins, 
            restindices < nrestbins, 
            restindices >= 0, 
            flux > 0, ivar > 0), axis=0)

        logFlux = np.log(flux[validbins])
        if len(logFlux) <= 0:
            continue
        restSlice = restindices[validbins]
        obsSlice = obsindices[validbins]#np.arange(offset,offset+combined.nPixels)[validbins]

        # calculate weights
        if args.unweighted:
            weights = None
        else:
            weights = ivar[validbins]#**(2)/flux[validbins]**4

        # Add this observation to our fitter
        fitter.addObservation(logFlux, obsSlice, restSlice, weights)
        fitTargets.append(target)
        npixels.append(len(logFlux))

    # Add constraint for continuum normalization
    if args.restnorm > 0:
        normCMin = args.restnorm-args.drestnorm/2
        normCMax = args.restnorm+args.drestnorm/2
        normCRange = np.arange(np.argmax(restWaveCenters > normCMin), 
            np.argmax(restWaveCenters > normCMax))
        normCCoefs = np.ones(len(normCRange))/len(normCRange)
        fitter.addConstraint('C', 0, normCRange, args.restnormweight*normCCoefs)
        if args.verbose:
            print 'Adding constraint: logC([%.4f,%.4f]) = %.1f (range covers %d continuum bins [%d,%d])' % (
                restWaveCenters[normCRange[0]], restWaveCenters[normCRange[-1]], 0, len(normCCoefs), normCRange[0], normCRange[-1])

    # Add constraint for transmission normalization
    if args.obsnorm > 0:
        normTMin = args.obsnorm-args.dobsnorm/2
        normTMax = args.obsnorm+args.dobsnorm/2
        normTRange = np.arange(np.argmax(transWaveCenters > normTMin), 
            np.argmax(transWaveCenters > normTMax))
        normTCoefs = np.ones(len(normTRange))/len(normTRange)
        fitter.addConstraint('T', 0, normTCoefs, args.obsnormweight*normTCoefs)
        if args.verbose:
            print 'Adding constraint: logT([%.4f,%.4f]) = %.1f (range covers %d transmission bins [%d,%d])' % (
                transWaveCenters[normTRange[0]], transWaveCenters[normTRange[-1]], 0, len(normTCoefs), normTRange[0], normTRange[-1])
    # run the fitter
    results = fitter.fit(verbose=args.verbose, atol=args.atol, btol=args.btol, max_iter=args.max_iter, sklearn=args.sklearn)
    chisq = fitter.getChiSquare()

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

    outfile.create_dataset('transWaveCenters', data=transWaveCenters)
    outfile.create_dataset('restWaveCenters', data=restWaveCenters)

    outfile.create_dataset('targets', data=[str(target) for target in fitTargets])
    outfile.create_dataset('redshifts', data=[target.z for target in fitTargets])

    outfile.create_dataset('T', data=obsModelValues)
    outfile.create_dataset('C', data=restModelValues)
    outfile.create_dataset('A', data=targetModelValues)

    if args.alpha:
        outfile.create_dataset('alpha', data=alphaModelValues)

    outfile.close()

if __name__ == '__main__':
    main()