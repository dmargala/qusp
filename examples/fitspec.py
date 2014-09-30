#!/usr/bin/env python
import argparse
import os

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

import bosslya
import random

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
    # fit options
    parser.add_argument("--sklearn", action="store_true",
        help="use sklearn linear regression instead of scipy lstsq")
    # scipy specifc options
    parser.add_argument("--max-iter", type=int, default=100,
        help="max number of iterations to use in lsqr")
    parser.add_argument("--atol", type=float, default=1e-4,
        help="a stopping tolerance")
    parser.add_argument("--btol", type=float, default=1e-8,
        help="b stopping tolerance")
    parser.add_argument("--z-col", type=int, default=3,
        help="redshift column of input targetlist")
    parser.add_argument("--norm-col", type=int, default=None,
        help="redshift column of input targetlist")
    parser.add_argument("--tilt-col", type=int, default=None,
        help="redshift column of input targetlist")
    bosslya.ContinuumModel.addArgs(parser)
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    fitsPath = os.path.join(boss_root, boss_version)

    fields = [('z',float,args.z_col)]
    if args.norm_col is not None:
        fields.append(('amp',float,args.norm_col))
    if args.tilt_col is not None:
        fields.append(('nu',float,args.tilt_col))

    # read target list
    targets = bosslya.target.readTargetList(args.input,fields)
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)

    # use the first n targets or a random sample
    if args.random:
        random.seed(args.seed)
        targets = random.sample(targets, ntargets)
    else:
        targets = targets[:ntargets]

    # we want to open the spPlate files in plate-mjd order
    targets = sorted(targets)

    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    # Initialize model 
    model = bosslya.ContinuumModel(**bosslya.ContinuumModel.fromArgs(args))

    if args.verbose:
        print '... adding observations to fit ...\n'

    # Add observations to model
    currentlyOpened = None
    fitTargets = []
    npixels = []
    for targetIndex, target in enumerate(targets):
        plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
        # load the spectrum file
        if plateFileName != currentlyOpened:
            if currentlyOpened is not None:
                spPlate.close()
            fullName = os.path.join(fitsPath,str(target.plate),plateFileName)
            # if args.verbose:
            #    print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)
            currentlyOpened = plateFileName

        # read this target's combined spectrum
        combined = bosslya.readCombinedSpectrum(spPlate, target.fiber)
        wavelength = combined.wavelength
        ivar = combined.ivar
        flux = combined.flux

        # Add this observation to our model
        nPixelsAdded = model.addObservation(target, flux, wavelength, ivar, 
            unweighted=args.unweighted)
        if nPixelsAdded > 0:
            fitTargets.append(target)
            npixels.append(nPixelsAdded)

    # Add constraints
    if args.restnormmax > args.restnormmin:
        model.addRestConstraint(0, args.restnormmin, args.restnormmax, args.restnormweight)
    if args.obsnormmax > args.obsnormmin:
        model.addObsConstraint(0, args.obsnormmin, args.obsnormmax, args.obsnormweight)
    if args.tiltweight > 0:
        model.addTiltConstraint(args.tiltweight)

    if args.verbose:
        print ''

    # run the fitter
    X, y = model.getModel()

    # perform fit
    if args.sklearn:
        from sklearn import linear_model
        regr = linear_model.LinearRegression(fit_intercept=False)
        if args.verbose:
            print '... performing fit using sklearn.linear_model.LinearRegression ...\n'
        regr.fit(X, y)
        soln = regr.coef_
    else:
        import scipy.sparse.linalg
        if args.verbose:
            print '... performing fit using scipy.sparse.linalg.lsqr ...\n'
        lsqr_soln = scipy.sparse.linalg.lsqr(X, y, show=args.verbose,
            iter_lim=args.max_iter, atol=args.atol, btol=args.btol)
        soln = lsqr_soln[0]

    chisq = model.getChiSq(soln)

    if args.verbose:
        print 'chisq (nModelParams,nConstraints): %.2g (%d,%d)' % (chisq, model.model.shape[1], model.nconstraints)
        print 'reduced chisq: %.2g' % (chisq/(model.model.shape[1]-model.nconstraints))

    # Save HDF5 file with results
    outfile = model.save(args.output+'.hdf5', soln, args)

    outfile.create_dataset('npixels', data=npixels)
    outfile.create_dataset('targets', data=[str(target) for target in fitTargets])
    outfile.create_dataset('redshifts', data=[target.z for target in fitTargets])

    try:
        sn = [target.sn for target in fitTargets]
    except AttributeError:
        sn = np.zeros(len(fitTargets))

    outfile.create_dataset('sn', data=sn)

    outfile.close()

    results = model.getResults(soln)
    with open(args.output+'.txt', 'w') as outfile:
        for i,target in enumerate(fitTargets):
            attrs = list(target.attrs())
            attrs.append(results['amplitude'][i])
            attrs.append(results['nu'][i])
            attrstr = ' '.join([str(attr) for attr in attrs])
            outfile.write('%s %s\n' % (str(target), attrstr))

if __name__ == '__main__':
    main()