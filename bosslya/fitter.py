import numpy as np

import scipy.sparse
import scipy.sparse.linalg

import bosslya

class ContinuumFitter():
    def __init__(self, params, obsWaveMin, obsWaveMax,
        restWaveMin, restWaveMax, restNParams, verbose=False):
        self.verbose = verbose

        # initialize binning arrays
        assert obsWaveMax > obsWaveMin, ('obsWaveMax must be greater than obsWaveMin')
        self.obsWaveMin = obsWaveMin
        self.obsWaveMax = obsWaveMax

        obsFiducialWave = bosslya.getFiducialWavelength(np.arange(4800))
        self.obsWaveMinIndex = np.argmax(obsFiducialWave > obsWaveMin)
        self.obsWaveMaxIndex = np.argmax(obsFiducialWave > obsWaveMax)+1

        self.obsWaveCenters = bosslya.getFiducialWavelength(np.arange(self.obsWaveMinIndex,self.obsWaveMaxIndex))
        self.obsNParams = len(self.obsWaveCenters)

        if verbose:
            print 'Observed frame bin centers span [%.2f:%.2f] with %d bins.' % (
                self.obsWaveCenters[0],self.obsWaveCenters[-1],self.obsNParams)

        assert restWaveMax > restWaveMin, ('restWaveMax must be greater than restWaveMin')
        self.restWaveMin = restWaveMin
        self.restWaveMax = restWaveMax
        self.restNParams = restNParams
        self.restWaveDelta = float(restWaveMax-restWaveMin)/restNParams
        self.restWaveCenters = 0.5*self.restWaveDelta + np.linspace(
            restWaveMin,restWaveMax,restNParams,endpoint=False)

        if verbose:
            print 'Rest frame bin centers span [%.2f:%.2f] with %d bins.' % (
                self.restWaveCenters[0],self.restWaveCenters[-1],self.restNParams)

        # the number of "model" pixels (excluding per target params)
        self.params = params
        self.nParams = len(params)
        self.nModelPixels = 0
        for param in params:
            if param['type'] is 'obs':
                self.nModelPixels += self.obsNParams
            elif param['type'] is 'rest':
                self.nModelPixels += self.restNParams
        # sparse matrix entry holders
        self.rowIndices = []
        self.colIndices = []
        self.coefficients = []
        self.logFluxes = []
        self.nTotalPixels = 0
        self.nTargets = 0
        self.soln = None
        self.nconstraints = 0

    def addObservation(self, target, flux, wavelength, ivar, unweighted=True):
        """
        Adds an observation to be fit. Each of logFlux, restSlice, restSlice should 
        have the same length. The obsSlice and restSlice specify the model pixel indices
        of the corresponding logFlux values.
        """

        # this spectrum's wavelength axis pixel offset
        offset = bosslya.getFiducialPixelIndexOffset(np.log10(wavelength[0]))

        obsFiducialIndices = np.arange(offset,offset+len(wavelength))
        obsFiducialWave = bosslya.getFiducialWavelength(obsFiducialIndices)

        restWave = obsFiducialWave/(1+target.z)
        restIndices = ((restWave - self.restWaveMin)/(self.restWaveMax - self.restWaveMin)*self.restNParams).astype(int)

        # trim ranges to valid data
        validbins = np.all((
            obsFiducialIndices >= self.obsWaveMinIndex, 
            obsFiducialIndices  < self.obsWaveMaxIndex, 
            restIndices < self.restNParams, 
            restIndices >= 0, 
            flux > 0, ivar > 0), axis=0)

        logFlux = np.log(flux[validbins])
        nPixels = len(logFlux)
        if nPixels <= 0:
            if self.verbose:
                print 'No good pixels in relavant range on target %s (z=%.2f)' % (target, target.z)
            return 0
        
        restIndices = restIndices[validbins]
        obsIndices = obsFiducialIndices[validbins]-self.obsWaveMinIndex

        assert np.amax(obsIndices) < self.obsNParams and np.amax(restIndices) < self.restNParams, (
            'Invalid model index value')

        # compute weights
        if unweighted:
            weights = np.ones(nPixels)
        else:
            weights = ivar[validbins]
        sqrtw = np.sqrt(weights)

        # Append logFlux values
        self.logFluxes.append(sqrtw*logFlux)

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
                colIndices.append(colOffset+obsIndices)
                colOffset += self.obsNParams
            elif param['type'] is 'rest':
                colIndices.append(colOffset+restIndices)
                colOffset += self.restNParams
            elif param['type'] is 'target':
                colIndices.append(colOffset+self.nTargets*np.ones(nPixels))
                colOffset += 1
        self.colIndices.append(np.concatenate(colIndices))

        # The coefficients in the model matrix are the sqrt(weight), unless a 'coef'
        # function is specified in the param dictionary
        for i,param in enumerate(self.params):
            if 'coef' in param.keys():
                coefficients.append(sqrtw*param['coef'](self.obsWaveCenters[obsIndices], self.restWaveCenters[restIndices]))
            else:
                coefficients.append(sqrtw)
        self.coefficients.append(np.concatenate(coefficients))

        # Increment the total number of pixel values and the number of observations
        self.nTotalPixels += nPixels
        self.nTargets += 1
        return nPixels

    def addConstraint(self, paramName, logFlux, wave, dwave, weight):
        waveMin = wave - 0.5*dwave
        waveMax = wave + 0.5*dwave
        if paramName is 'T':
            waves = self.obsWaveCenters
        elif paramName is 'C':
            waves = self.restWaveCenters
        else:
            assert False, ('Invalid constraint parameter')

        waveIndexRange = np.arange(np.argmax(waves > waveMin), np.argmax(waves > waveMax))
        normCoefs = weight*np.ones(len(waveIndexRange))/len(waveIndexRange)

        if self.verbose:
            print 'Adding constraint: log%s([%.4f:%.4f]) = %.1f (range covers %d bins [%d:%d])' % (
                paramName, waves[waveIndexRange[0]], waves[waveIndexRange[-1]], logFlux, 
                len(normCoefs), waveIndexRange[0], waveIndexRange[-1])

        colOffset = 0
        for i,param in enumerate(self.params):
            if param['name'] is paramName:
                self.colIndices.append(colOffset+waveIndexRange)
            if param['type'] is 'obs':
                colOffset += self.obsNParams
            elif param['type'] is 'rest':
                colOffset += self.restNParams
            elif param['type'] is 'target':
                colOffset += 1

        self.coefficients.append(normCoefs)
        self.rowIndices.append(self.nTotalPixels*np.ones(len(waveIndexRange)))
        self.logFluxes.append([logFlux])

        self.nTotalPixels += 1
        self.nconstraints += 1        

    def fit(self, atol=1e-8, btol=1e-8, max_iter=100, sklearn=False):
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
        self.model = model.tocsc()

        if self.verbose:
            print 'Number of transmission model params: %d' % self.obsNParams
            print 'Number of continuum model params: %d' % self.restNParams
            print 'Number of targets: %d' % self.nTargets
            print 'Total model params: %d' % nModelPixels
            print 'Total number of flux measurements: %d' % self.nTotalPixels
            print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)\n' % (
                self.model.data.nbytes, self.model.indptr.nbytes, self.model.indices.nbytes)

        # perform fit
        if sklearn:
            from sklearn import linear_model
            regr = linear_model.LinearRegression()
            if self.verbose:
                print '... performing fit using sklearn.linear_model.LinearRegression ...\n'
            regr.fit(self.model, logFluxes)
            self.soln = regr.coef_
        else:
            if self.verbose:
                print '... sperforming fit using cipy.sparse.linalg.lsqr ...\n'
            soln = scipy.sparse.linalg.lsqr(self.model, logFluxes, show=self.verbose,
                iter_lim=max_iter, atol=atol, btol=btol)
            self.soln = soln[0]

        # return results
        return self.getResults()

    def getResults(self):
        """
        Returns a dictionary containing fit results
        """
        assert self.soln is not None, ('Can\'t request results before fitting')
        results = {}
        offset = 0
        for i,param in enumerate(self.params):
            if param['type'] is 'obs':
                npixels = self.obsNParams
            elif param['type'] is 'rest':
                npixels = self.restNParams
            elif param['type'] is 'target':
                npixels = self.nTargets
            results[param['name']] = self.soln[offset:offset+npixels]
            offset += npixels
        return results

    def getChiSq(self):
        """
        Returns chisq of best fit
        """
        assert self.soln is not None, ('Can\'t request results before fitting')
        logFluxes = np.concatenate(self.logFluxes)
        residuals = logFluxes - self.model.dot(self.soln)
        return np.dot(residuals,residuals)/len(residuals)

    def getObservationChiSq(self, i):
        """
        Returns chisq specified observation i
        """
        nModelPixels = self.nModelPixels + self.nTargets

        logFluxes = self.logFluxes[i]
        nPixels = len(logFluxes)

        rowIndices = np.tile(np.arange(nPixels), self.nParams)
        colIndices = self.colIndices[i]
        coefficients = self.coefficients[i]

        # build the sparse matrix
        model = scipy.sparse.coo_matrix((coefficients,(rowIndices,colIndices)), 
            shape=(nPixels,nModelPixels), dtype=np.float32)
        # convert the sparse matrix to compressed sparse column format
        model = model.tocsc()

        # calculate chisq
        residuals = logFluxes - model.dot(self.soln)
        return np.dot(residuals, residuals)/len(residuals)


    @staticmethod
    def addArgs(parser):
        # transmission model wavelength grid options
        parser.add_argument("--obsmin", type=float, default=3600,
            help="transmission model wavelength minimum")
        parser.add_argument("--obsmax", type=float, default=10000,
            help="transmission model wavelength maximum")
        # continuum model wavelength grid options
        parser.add_argument("--restmin", type=float, default=850,
            help="rest wavelength minimum")
        parser.add_argument("--restmax", type=float, default=2850,
            help="rest wavelength maximum")
        parser.add_argument("--nrestbins", type=int, default=500,
            help="number of restframe bins")
        ####### constraints ########
        parser.add_argument("--restnorm", type=float, default=1280,
            help="restframe wavelength to normalize at")
        parser.add_argument("--drestnorm", type=float, default=10,
            help="restframe window size +/- on each side of restnorm wavelength")
        parser.add_argument("--restnormweight", type=float, default=1e4,
            help="norm constraint weight")
        parser.add_argument("--obsnorm", type=float, default=5000,
            help="obsframe wavelength to normalize at")
        parser.add_argument("--dobsnorm", type=float, default=10,
            help="obsframe window size +/- on each side of obsnorm wavelength")
        parser.add_argument("--obsnormweight", type=float, default=1e4,
            help="norm constraint weight")
        ####### fit options #######
        parser.add_argument("--unweighted", action="store_true",
        help="perform unweighted least squares fit")
        parser.add_argument("--sklearn", action="store_true",
            help="use sklearn linear regression instead of scipy lstsq")
        ## scipy fit options
        parser.add_argument("--max-iter", type=int, default=100,
            help="max number of iterations to use in lsqr")
        parser.add_argument("--atol", type=float, default=1e-4,
            help="a stopping tolerance")
        parser.add_argument("--btol", type=float, default=1e-8,
            help="b stopping tolerance")