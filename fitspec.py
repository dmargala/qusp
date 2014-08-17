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

class QuasarModel():
    def __init__(self, params, obsWaveCenters, restWaveCenters):
        self.params = params
        self.nParams = len(params)
        self.obsWaveCenters = obsWaveCenters
        self.nObs = len(obsWaveCenters)
        self.restWaveCenters = restWaveCenters
        self.nRest = len(restWaveCenters)
        self.rowIndices = []
        self.colIndices = []
        self.coefficients = []
        self.logFluxes = []
        self.nTotalPixels = 0
        self.nTargets = 0
        self.nModelPixels = 0
        for name, param in params.items():
            if param['type'] is 'obs':
                self.nModelPixels += self.nObs
            elif param['type'] is 'rest':
                self.nModelPixels += self.nRest
        print self.nModelPixels

    def addQuasar(self, logFlux, obsSlice, restSlice):
        nPixels = len(logFlux)

        self.logFluxes.append(logFlux)

        fluxIndices = self.nTotalPixels+np.arange(nPixels)
        self.rowIndices.append(np.tile(fluxIndices,self.nParams))

        colIndices = []
        coefficients = []
        offset = 0
        for i,name in enumerate(self.params.keys()):
            if self.params[name]['type'] is 'obs':
                coefs = np.ones(nPixels)
                colIndices.append(offset+obsSlice)
                offset += self.nObs
            elif self.params[name]['type'] is 'rest':
                if 'coef' in self.params[name].keys():
                    coefs = self.params[name][coef](self.obsWaveCenters[obsSlice], self.restWaveCenters[restSlice])
                else:
                    coefs = np.ones(nPixels)
                colIndices.append(offset+restSlice)
                offset += self.nRest
            elif self.params[name]['type'] is 'target':
                coefs = np.ones(nPixels)
                colIndices.append(offset+self.nTargets*np.ones(nPixels))
                offset += 1
            coefficients.append(coefs)
        self.colIndices.append(np.concatenate(colIndices))
        self.coefficients.append(np.concatenate(coefficients))

        self.nTargets += 1
        self.nTotalPixels += nPixels

    def fit(self, atol=1e-8, btol=1e-8, max_iter=100, verbose=False, sklearn=False):
        self.nModelPixels += self.nTargets

        rowIndices = np.concatenate(self.rowIndices)
        colIndices = np.concatenate(self.colIndices)
        coefficients = np.concatenate(self.coefficients)
        logFluxes = np.concatenate(self.logFluxes)

        model = scipy.sparse.coo_matrix((coefficients,(rowIndices,colIndices)), 
            shape=(self.nTotalPixels,self.nModelPixels), dtype=np.float32)
        model = model.tocsc()

        if verbose:
            print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)' % (model.data.nbytes, model.indptr.nbytes, model.indices.nbytes)

        if sklearn:
            from sklearn import linear_model
            regr = linear_model.LinearRegression()
            regr.fit(model, logFluxes)
            self.soln = regr.coef_
        else:
            soln = scipy.sparse.linalg.lsqr(model, logFluxes, show=verbose, iter_lim=max_iter, atol=atol, btol=btol)
            self.soln = soln[0]

    def getParams(self):
        results = {}
        offset = 0
        for i,name in enumerate(self.params.keys()):
            if self.params[name]['type'] is 'obs':
                npixels = self.nObs
            elif self.params[name]['type'] is 'rest':
                npixels = self.nRest
            elif self.params[name]['type'] is 'target':
                npixels = self.nTargets
            results[name] = self.soln[offset:offset+npixels]
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

    restmin = 850
    restmax = 1850
    nrestbins = 250
    drest = float(restmax-restmin)/nrestbins
    restWaveCenters = np.linspace(restmin,restmax,nrestbins) + drest/2


    params = {}
    params['T'] = {'type':'obs'}
    params['C'] = {'type':'rest'}
    params['A'] = {'type':'target'}

    if args.alpha:
        def alphaCoef(obs, rest):
            return -np.power(obs/rest, args.beta)
        params['alpha'] = {'type':'rest','coef':alphaCoef}

    model = QuasarModel(params, obsWaveCenters, restWaveCenters)

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

        model.addQuasar(logFlux, obsSlice, restSlice)

    model.fit(verbose=args.verbose, atol=args.atol, btol=args.btol, max_iter=args.max_iter, sklearn=args.sklearn)
    results = model.getParams()

    obsModelValues = np.exp(results['T'])
    restModelValues = np.exp(results['C'])
    targetModelValues = np.exp(results['A'])

    if args.alpha:
        alphaModelValues = results['alpha']

    # Save HDF5 file with results
    outfile = h5py.File(args.output,'w')

    outfile.create_dataset('obsWaveCenters', data=obsWaveCenters)
    outfile.create_dataset('restWaveCenters', data=restWaveCenters)

    outfile.create_dataset('T', data=obsModelValues)
    outfile.create_dataset('C', data=restModelValues)
    outfile.create_dataset('A', data=targetModelValues)

    if args.alpha:
        outfile.create_dataset('alpha', data=alphaModelValues)

    outfile.create_dataset('targets', data=[str(target) for target in targets])
    outfile.create_dataset('redshifts', data=[target.z for target in targets])

    outfile.close()

    fig = plt.figure()

if __name__ == '__main__':
    main()