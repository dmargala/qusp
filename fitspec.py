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

class ContinuumFitter():
    def __init__(self, params, obsWaveCenters, restWaveCenters):
        self.params = params
        self.nParams = len(params)
        self.obsWaveCenters = obsWaveCenters
        self.nObs = len(obsWaveCenters)
        self.restWaveCenters = restWaveCenters
        self.nRest = len(restWaveCenters)
        # the number of "model" pixels (excluding per target params)
        self.nModelPixels = 0
        for param in params:
            if param['type'] is 'obs':
                self.nModelPixels += self.nObs
            elif param['type'] is 'rest':
                self.nModelPixels += self.nRest
        # sparse matrix entry holders
        self.rowIndices = []
        self.colIndices = []
        self.coefficients = []
        self.logFluxes = []
        self.nTotalPixels = 0
        self.nTargets = 0

    def addObservation(self, logFlux, obsSlice, restSlice):
        """
        Adds an observation to be fit. Each of logFlux, restSlice, restSlice should 
        have the same length. The obsSlice and restSlice specify the model pixel indices
        of the corresponding logFlux values.
        """
        nPixels = len(logFlux)
        assert len(logFlux) > 0, ('Empty flux array')
        assert len(obsSlice) == nPixels and len(restSlice) == nPixels, (
            'Input data array sizes do not match')
        assert np.amax(obsSlice) < self.nObs and np.amax(restSlice) < self.nRest, (
            'Invalid model index value')

        # Save logFlux values
        self.logFluxes.append(logFlux)

        # Each row corresponds to single flux value, the model matrix
        # will have nParams entries per row
        rowIndices = self.nTotalPixels+np.arange(nPixels)
        self.rowIndices.append(np.tile(rowIndices,self.nParams))

        # Each col corresponds to model parameter value, the model matrix
        # is ordered in blocks of model parameters
        colIndices = []
        colOffset = 0
        coefficients = []
        for i,param in enumerate(self.params):
            if param['type'] is 'obs':
                colIndices.append(colOffset+obsSlice)
                colOffset += self.nObs
            elif param['type'] is 'rest':
                colIndices.append(colOffset+restSlice)
                colOffset += self.nRest
            elif param['type'] is 'target':
                colIndices.append(colOffset+self.nTargets*np.ones(nPixels))
                colOffset += 1
        self.colIndices.append(np.concatenate(colIndices))

        # The coefficients in the model matrix are 1, unless a 'coef'
        # function is specified in the param dictionary
        for i,param in enumerate(self.params):
            if 'coef' in param.keys():
                coefficients.append(param[coef](self.obsWaveCenters[obsSlice], self.restWaveCenters[restSlice]))
            else:
                coefficients.append(np.ones(nPixels))
        self.coefficients.append(np.concatenate(coefficients))

        # Increment the total number of pixel values and the number of observations
        self.nTotalPixels += nPixels
        self.nTargets += 1

    def fit(self, atol=1e-8, btol=1e-8, max_iter=100, sklearn=False, verbose=False):
        """
        Does final assembly of the sparse matrix representing the model and performs least
        squares fit.
        """
        nModelPixels = self.nModelPixels + self.nTargets

        rowIndices = np.concatenate(self.rowIndices)
        colIndices = np.concatenate(self.colIndices)
        coefficients = np.concatenate(self.coefficients)
        logFluxes = np.concatenate(self.logFluxes)

        # build the sparse matrix
        model = scipy.sparse.coo_matrix((coefficients,(rowIndices,colIndices)), 
            shape=(self.nTotalPixels,nModelPixels), dtype=np.float32)
        # convert the sparse matrix to compressed sparse column format
        model = model.tocsc()

        if verbose:
            print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)' % (model.data.nbytes, model.indptr.nbytes, model.indices.nbytes)

        # perform fit
        if sklearn:
            from sklearn import linear_model
            regr = linear_model.LinearRegression()
            regr.fit(model, logFluxes)
            self.soln = regr.coef_
        else:
            soln = scipy.sparse.linalg.lsqr(model, logFluxes, show=verbose, iter_lim=max_iter, atol=atol, btol=btol)
            self.soln = soln[0]

        # return results
        return self.getParams()

    def getParams(self):
        results = {}
        offset = 0
        for i,param in enumerate(self.params):
            if param['type'] is 'obs':
                npixels = self.nObs
            elif param['type'] is 'rest':
                npixels = self.nRest
            elif param['type'] is 'target':
                npixels = self.nTargets
            results[param['name']] = self.soln[offset:offset+npixels]
            offset += npixels
        return results

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
    parser.add_argument("--atol", type=float, default=1e-4,
        help="a stopping tolerance")
    parser.add_argument("--btol", type=float, default=1e-8,
        help="b stopping tolerance")
    parser.add_argument("--sklearn", action="store_true",
        help="use sklearn linear regression")
    parser.add_argument("--beta", type=float, default=3.92,
        help="optical depth power law parameter")
    parser.add_argument("-o","--output", type=str, default=None,
        help = "output filename")
    parser.add_argument("--alpha", action="store_true",
        help = "add z evolution term")
    parser.add_argument("--nrestbins", type=int, default=250,
        help="number of restframe bins")
    parser.add_argument("--restmax", type=float, default=1850,
        help="rest wavelength minimum")
    parser.add_argument("--restmin", type=float, default=850,
        help="rest wavelength maximum")
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

    restmin = args.restmin
    restmax = args.restmax
    nrestbins = args.nrestbins
    drest = float(restmax-restmin)/nrestbins
    restWaveCenters = np.linspace(restmin,restmax,nrestbins,endpoint=False) + drest/2

    # Initialize model
    params = []
    params.append({'name':'T','type':'obs'})
    params.append({'name':'C','type':'rest'})
    params.append({'name':'A','type':'target'})

    if args.alpha:
        def alphaCoef(obs, rest):
            return -np.power(obs/rest, args.beta)
        params.append({'name':'alpha','type':'rest','coef':alphaCoef})

    model = ContinuumFitter(params, obsWaveCenters, restWaveCenters)

    # loop over targets
    plateFileName = None
    for targetIndex, target in enumerate(targets):

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

        obswave = obsWaveCenters[slice(offset,offset+combined.nPixels)]
        restwave = obswave/(1+target.z)
        restindices = ((restwave - restmin)/(restmax - restmin)*nrestbins).astype(int)

        validbins = np.all((restindices < nrestbins, restindices >= 0, flux > 0, ivar > 0), axis=0)

        logFlux = np.log(flux[validbins])
        restSlice = restindices[validbins]
        obsSlice = np.arange(offset,offset+combined.nPixels)[validbins]

        # Add this observation to our fitter
        model.addObservation(logFlux, obsSlice, restSlice)

    # run the fitter
    results = model.fit(verbose=args.verbose, atol=args.atol, btol=args.btol, max_iter=args.max_iter, sklearn=args.sklearn)

    obsModelValues = np.exp(results['T'])
    restModelValues = np.exp(results['C'])
    targetModelValues = np.exp(results['A'])

    if args.alpha:
        alphaModelValues = results['alpha']

    # Save HDF5 file with results
    outfile = h5py.File(args.output,'w')

    outfile.create_dataset('obsWaveCenters', data=obsWaveCenters)
    outfile.create_dataset('restWaveCenters', data=restWaveCenters)

    outfile.create_dataset('targets', data=[str(target) for target in targets])
    outfile.create_dataset('redshifts', data=[target.z for target in targets])

    outfile.create_dataset('T', data=obsModelValues)
    outfile.create_dataset('C', data=restModelValues)
    outfile.create_dataset('A', data=targetModelValues)

    if args.alpha:
        outfile.create_dataset('alpha', data=alphaModelValues)

    outfile.close()

if __name__ == '__main__':
    main()