#!/usr/bin/env python
import argparse
import os

import numpy as np
import h5py
from astropy.io import fits

import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

import matplotlib.pyplot as plt

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
    parser.add_argument("--boss-version", type=str, default="v5_7_0",
        help = "boss pipeline version tag")
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("--max-iter", type=int, default=100,
        help="max number of iterations to use in lsqr")
    parser.add_argument("--atol", type=float, default=1e-2,
        help="a stopping tolerance")
    parser.add_argument("--btol", type=float, default=1e-8,
        help="b stopping tolerance")
    parser.add_argument("--sklearn", action="store_true",
        help="use sklearn linear regression")
    parser.add_argument("--beta", type=float, default=3.92,
        help="optical depth power law parameter")
    parser.add_argument("-o","--output", type=str, default=None,
        help = "output filename")
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

    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    # initialize binning arrays
    nobsbins = 4800
    obswavecenters = bosslya.getFiducialWavelength(np.arange(nobsbins))

    restmin = 850
    restmax = 1850
    nrestbins = 250
    drest = float(restmax-restmin)/nrestbins
    restwavecenters = np.linspace(restmin,restmax,nrestbins) + drest/2

    modelRowIndices = []
    modelColIndices = []
    modelCoefficients = []

    logFluxes = []

    nTotalPixels = 0

    nModelPixels = nobsbins + 2*nrestbins + ntargets

    # loop over targets
    plateFileName = None
    for targetIndex, target in enumerate(targets[:ntargets]):

        # load the spectrum file
        if plateFileName != 'spPlate-%s-%s.fits' % (target.plate, target.mjd):
            plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
            fullName = os.path.join(fitsPath,str(target.plate),plateFileName)
            if args.verbose:
                print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)

        # read this target's combined spectrum
        combined = bosslya.readCombinedSpectrum(spPlate, target.fiber)
        wavelength = combined.wavelength
        ivar = combined.ivar
        flux = combined.flux

        # this spectrum's wavelength axis pixel offset
        offset = bosslya.getFiducialPixelIndexOffset(np.log10(wavelength[0]))

        obswave = obswavecenters[slice(offset,offset+combined.nPixels)]
        restwave = obswave/(1+target.z)
        restindices = ((restwave - restmin)/(restmax - restmin)*nrestbins).astype(int)

        validbins = np.all((restindices < nrestbins, restindices >= 0, flux > 0, ivar > 0), axis=0)

        logFlux = np.log(flux[validbins])
        restSlice = restindices[validbins]
        obsSlice = np.arange(offset,offset+combined.nPixels)[validbins]

        # Build sparse matrix representation of model
        nPixels = len(logFlux)
        logFluxes.append(logFlux)

        normSlice = targetIndex*np.ones(nPixels)

        fluxIndices = nTotalPixels+np.arange(nPixels)
        modelRowIndices.append(np.concatenate([fluxIndices,fluxIndices,fluxIndices,fluxIndices]))
        modelColIndices.append(np.concatenate([obsSlice,nobsbins+restSlice,nobsbins+nrestbins+restSlice,nobsbins+2*nrestbins+normSlice]))

        beta = args.beta
        alphaCoefs = -np.power(obswavecenters[obsSlice]/(restwavecenters[restSlice]),beta)

        modelCoefficients.append(np.concatenate([np.ones(2*nPixels),alphaCoefs,np.ones(nPixels)]))

        nTotalPixels += nPixels

    modelRowIndices = np.concatenate(modelRowIndices)
    modelColIndices = np.concatenate(modelColIndices)
    modelCoefficients = np.concatenate(modelCoefficients)
    logFluxVector = np.concatenate(logFluxes)

    modelMatrix = scipy.sparse.coo_matrix((modelCoefficients,(modelRowIndices,modelColIndices)), 
        shape=(nTotalPixels,nModelPixels), dtype=np.float32)
    modelMatrix = modelMatrix.tocsc()

    if args.verbose:
        print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)' % (modelMatrix.data.nbytes, modelMatrix.indptr.nbytes, modelMatrix.indices.nbytes)

    if args.sklearn:
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        regr.fit(modelMatrix, logFluxVector)
        coef = regr.coef_
    else:
        soln = scipy.sparse.linalg.lsqr(modelMatrix, logFluxVector, show=args.verbose, iter_lim=args.max_iter, atol=args.atol, btol=args.btol)
        coef = soln[0]

    obsModelValues = np.exp(coef[:nobsbins])
    restModelValues = np.exp(coef[nobsbins:nobsbins+nrestbins])
    alphaModelValues = coef[nobsbins+nrestbins:nobsbins+2*nrestbins]
    targetModelValues = np.exp(coef[nobsbins+2*nrestbins:])

    # Save HDF5 file with results
    outfile = h5py.File(args.output,'w')

    outfile.create_dataset('obsWaveCenters', data=obswavecenters)
    outfile.create_dataset('restWaveCenters', data=restwavecenters)

    outfile.create_dataset('T', data=obsModelValues)
    outfile.create_dataset('C', data=restModelValues)
    outfile.create_dataset('alpha', data=alphaModelValues)
    outfile.create_dataset('A', data=targetModelValues)

    outfile.create_dataset('targets', data=[str(target) for target in targets[:ntargets]])
    outfile.create_dataset('redshifts', data=[target.z for target in targets[:ntargets]])

    outfile.close()

    fig = plt.figure()

if __name__ == '__main__':
    main()