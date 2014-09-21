import numpy as np

import scipy.sparse
import scipy.sparse.linalg

import bosslya

class ContinuumFitter():
    def __init__(self, obsWaveMin, obsWaveMax,
        restWaveMin, restWaveMax, restNParams, nuWave=0,
        alphaMin=1025, alphaMax=1216, beta=3.92, verbose=False):
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

        self.beta = beta
        self.alphaMin = max(alphaMin,restWaveMin)
        self.alphaMax = min(alphaMax,restWaveMax)
        self.alphaMinIndex = np.argmax(self.restWaveCenters >= self.alphaMin)
        self.alphaMaxIndex = np.argmax(self.restWaveCenters > self.alphaMax)
        self.alphaWaveCenters = self.restWaveCenters[self.alphaMinIndex:self.alphaMaxIndex]
        self.alphaNParams = len(self.alphaWaveCenters)

        if verbose:
            if self.alphaNParams > 0:
                print 'Absorption bin centers span [%.2f:%.2f] with %d bins.' % (
                    self.alphaWaveCenters[0], self.alphaWaveCenters[-1], self.alphaNParams)
            else:
                print 'No absorption params'

        self.targetNParams = 1

        self.nuWave = nuWave
        if nuWave > 0:
            self.targetNParams += 1
        # the number of "model" pixels (excluding per target params)
        self.nModelPixels = self.obsNParams + self.restNParams + self.alphaNParams
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

        logFlux = np.log(flux[validbins]) + np.log(1+target.z)
        nPixels = len(logFlux)
        if nPixels <= 0:
            if self.verbose:
                print 'No good pixels in relavant range on target %s (z=%.2f)' % (target, target.z)
            return 0
    
        # compute weights
        if unweighted:
            weights = np.ones(nPixels)
        else:
            weights = ivar[validbins]
        sqrtw = np.sqrt(weights)

        # Append logFlux values
        logFluxes = sqrtw*logFlux

        # Assemble matrix
        colIndices = []
        rowIndices = []
        coefficients = []
        rowOffset = self.nTotalPixels
        colOffset = 0

        def buildBlock(rows, cols, paramValues):
            # Each col corresponds to model parameter value, the model matrix
            # is ordered in blocks of model parameters
            colIndices.append(colOffset + cols)
            # Each row corresponds to single flux value, the model matrix
            # will have nParams entries per row
            rowIndices.append(rowOffset + rows)
            # The coefficients in the model matrix are the sqrt(weight), unless a 'coef'
            # function is specified in the param dictionary
            coefficients.append(paramValues)

        obsIndices = obsFiducialIndices[validbins]-self.obsWaveMinIndex
        assert np.amax(obsIndices) < self.obsNParams, (
            'Invalid obsmodel index value')

        buildBlock(np.arange(nPixels), obsIndices, np.ones(nPixels))
        colOffset += self.obsNParams

        restIndices = restIndices[validbins]
        assert np.amax(restIndices) < self.restNParams, (
            'Invalid rest model index value')

        buildBlock(np.arange(nPixels), restIndices, np.ones(nPixels))
        colOffset += self.restNParams

        alphaMinIndex = np.argmax(restIndices == self.alphaMinIndex)
        alphaMaxIndex = np.argmax(restIndices == self.alphaMaxIndex)

        if alphaMaxIndex > alphaMinIndex:
            alphaRows = np.arange(nPixels)[alphaMinIndex:alphaMaxIndex]
            alphaIndices = restIndices[alphaMinIndex:alphaMaxIndex] - self.alphaMinIndex

            assert np.amax(alphaIndices) < self.alphaNParams, 'Invalid alpha index value'
            alphaValues = -np.ones(len(alphaIndices))*np.power(1+target.z,self.beta)

            buildBlock(alphaRows, alphaIndices, alphaValues)
        colOffset += self.alphaNParams

        targetIndices = self.targetNParams*self.nTargets*np.ones(nPixels)

        buildBlock(np.arange(nPixels), targetIndices, np.ones(nPixels))
        colOffset += 1

        if self.nuWave > 0:
            buildBlock(np.arange(nPixels), targetIndices, np.log(self.restWaveCenters[restIndices]/self.nuWave))
            colOffset += 1

        self.addModelCoefficents(np.concatenate(rowIndices),
            np.concatenate(colIndices), np.concatenate(coefficients), logFluxes)
        self.nTargets += 1

        return nPixels

    def addRestConstraint(self, logFlux, wavemin, wavemax, weight):
        waves = self.restWaveCenters
        offset = self.obsNParams

        waveIndexRange = np.arange(np.argmax(waves > wavemin), np.argmax(waves > wavemax))
        constraintCoefficients = weight*np.ones(len(waveIndexRange))

        if self.verbose:
            print 'Adding constraint: sum(%.2g*logC([%.2f:%.2f])) = %.1f (%d logC params [%d:%d])' % (
                weight, waves[waveIndexRange[0]], waves[waveIndexRange[-1]], logFlux, 
                len(waveIndexRange), waveIndexRange[0], waveIndexRange[-1])

        colIndices = offset+waveIndexRange
        rowIndices = self.nTotalPixels*np.ones(len(constraintCoefficients))

        self.addModelCoefficents(rowIndices, colIndices, constraintCoefficients, [logFlux])
        self.nconstraints += 1

    def addObsConstraint(self, logFlux, wavemin, wavemax, weight):
        waves = self.obsWaveCenters
        offset = 0

        waveIndexRange = np.arange(np.argmax(waves > wavemin), np.argmax(waves > wavemax)+1)
        nconstraints = len(waveIndexRange)

        constraintCoefficients = weight*np.ones(nconstraints)

        if self.verbose:
            print 'Adding constraint: %.2g*logT([%.2f:%.2f]) = %.1f (%d logT params [%d:%d])' % (
                weight, waves[waveIndexRange[0]], waves[waveIndexRange[-1]], logFlux, 
                len(waveIndexRange), waveIndexRange[0], waveIndexRange[-1])

        colIndices = offset+waveIndexRange
        rowIndices = self.nTotalPixels+np.arange(nconstraints)

        logFluxes = logFlux*np.ones(nconstraints)

        self.addModelCoefficents(rowIndices, colIndices, constraintCoefficients, logFluxes)
        self.nconstraints += nconstraints

    def addNuConstraint(self, weight):
        
        colIndices = 1 + self.nModelPixels + np.arange(0,self.targetNParams*self.nTargets,self.targetNParams)

        assert len(colIndices) == self.nTargets, ('Invalid number of nu params')

        if self.verbose:
            print 'Adding constraint: sum(nu) = 0 (%d nu params)' % self.nTargets

        rowIndices = self.nTotalPixels*np.ones(self.nTargets)
        constraintCoefficients = weight*np.ones(self.nTargets)/self.nTargets

        self.addModelCoefficents(rowIndices, colIndices, constraintCoefficients, [0])
        self.nconstraints += 1

    def addModelCoefficents(self, rows, cols, coefs, logFluxes):
        self.colIndices.append(cols)
        self.rowIndices.append(rows)
        self.coefficients.append(coefs)
        self.logFluxes.append(logFluxes)
        self.nTotalPixels += len(logFluxes)

    def fit(self, atol=1e-8, btol=1e-8, max_iter=100, sklearn=False):
        """
        Does final assembly of the sparse matrix representing the model and performs least
        squares fit.
        """
        nModelPixels = self.nModelPixels + self.targetNParams*self.nTargets

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
            print 'Number of absorption model params: %d' % self.alphaNParams
            print 'Number of targets: %d' % self.nTargets
            print 'Number of target params: %d' % self.targetNParams
            print ''
            print 'Total number of model params: %d' % nModelPixels
            print 'Total number of flux measurements: %d (%d constraints)' % (self.nTotalPixels, self.nconstraints)
            print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)\n' % (
                self.model.data.nbytes, self.model.indptr.nbytes, self.model.indices.nbytes)

        # perform fit
        if sklearn:
            from sklearn import linear_model
            regr = linear_model.LinearRegression(fit_intercept=False)
            if self.verbose:
                print '... performing fit using sklearn.linear_model.LinearRegression ...\n'
            regr.fit(self.model, logFluxes)
            self.soln = regr.coef_
        else:
            if self.verbose:
                print '... performing fit using scipy.sparse.linalg.lsqr ...\n'
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

        results['T'] = self.soln[offset:offset+self.obsNParams]
        offset += self.obsNParams
        results['C'] = self.soln[offset:offset+self.restNParams]
        offset += self.restNParams
        results['alpha'] = self.soln[offset:offset+self.alphaNParams]
        offset += self.alphaNParams
        results['A'] = self.soln[offset:offset+self.targetNParams*self.nTargets:self.targetNParams]
        offset += 1
        if self.nuWave > 0:
            results['nu'] = self.soln[offset:offset+self.targetNParams*self.nTargets:self.targetNParams]
            offset += 1

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
        Returns chisq of the specified observation index
        """
        nModelPixels = self.nModelPixels + self.targetNParams*self.nTargets

        logFluxes = self.logFluxes[i]
        nPixels = len(logFluxes)

        rowIndices = self.rowIndices[i] - min(self.rowIndices[i])
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

    def save(self, outfile, args):
        outfile.create_dataset('model_data', data=self.model.data)
        outfile.create_dataset('model_indices', data=self.model.indices)
        outfile.create_dataset('model_indptr', data=self.model.indptr)
        outfile.create_dataset('model_shape', data=self.model.shape)
        outfile.create_dataset('soln', data=self.soln)

        dsetObsWave = outfile.create_dataset('obsWaveCenters', data=self.obsWaveCenters)
        dsetRestWave = outfile.create_dataset('restWaveCenters', data=self.restWaveCenters)

        # transform fit results
        results = self.getResults()
        obsModelValues = np.exp(results['T'])
        restModelValues = np.exp(results['C'])
        targetModelValues = np.exp(results['A'])
        alphaModelValues = results['alpha']
        nuModelValues = results['nu']

        dsetT = outfile.create_dataset('T', data=obsModelValues)
        dsetT.attrs['normmin'] = args.obsnormmin
        dsetT.attrs['normmax'] = args.obsnormmax
        dsetT.attrs['normweight'] = args.obsnormweight

        dsetC = outfile.create_dataset('C', data=restModelValues)
        dsetC.attrs['normmin'] = args.restnormmin
        dsetC.attrs['normmax'] = args.restnormmax
        dsetC.attrs['normweight'] = args.restnormweight

        dsetA = outfile.create_dataset('A', data=targetModelValues)

        dsetAlpha = outfile.create_dataset('alpha', data=alphaModelValues)
        dsetAlpha.attrs['minRestIndex'] = self.alphaMinIndex
        dsetAlpha.attrs['maxRestIndex'] = self.alphaMaxIndex 
        dsetAlpha.attrs['beta'] = self.beta

        dsetNu = outfile.create_dataset('nu', data=nuModelValues)
        dsetNu.attrs['nuwave'] = args.nuwave
        dsetNu.attrs['normweight'] = args.nuweight
    
        chiSqs = [self.getObservationChiSq(i) for i in range(self.nTargets)]
        outfile.create_dataset('chisq', data=chiSqs)


    @staticmethod
    def addArgs(parser):
        # transmission model wavelength grid options
        parser.add_argument("--obsmin", type=float, default=3600,
            help="transmission model wavelength minimum")
        parser.add_argument("--obsmax", type=float, default=10000,
            help="transmission model wavelength maximum")
        parser.add_argument("--obsnormmin", type=float, default=3600,
            help="obsframe wavelength to normalize at")
        parser.add_argument("--obsnormmax", type=float, default=10000,
            help="obsframe window size +/- on each side of obsnorm wavelength")
        parser.add_argument("--obsnormweight", type=float, default=1e3,
            help="norm constraint weight")
        # continuum model wavelength grid options
        parser.add_argument("--restmin", type=float, default=850,
            help="rest wavelength minimum")
        parser.add_argument("--restmax", type=float, default=2850,
            help="rest wavelength maximum")
        parser.add_argument("--nrestbins", type=int, default=500,
            help="number of restframe bins")
        parser.add_argument("--restnormmin", type=float, default=1275,
            help="restframe wavelength to normalize at")
        parser.add_argument("--restnormmax", type=float, default=1285,
            help="restframe window size +/- on each side of restnorm wavelength")
        parser.add_argument("--restnormweight", type=float, default=1e3,
            help="norm constraint weight")
        # absorption model parameter options
        parser.add_argument("--alphamin", type=float,default=1025,
            help="alpha min wavelength")
        parser.add_argument("--alphamax", type=float,default=1216,
            help="alpha max wavelength")
        parser.add_argument("--beta", type=float, default=3.92,
            help="optical depth power law parameter")
        # spectral tilt parameter options
        parser.add_argument("--nuwave", type=float, default=1480,
            help="spectral tilt wavelength")
        parser.add_argument('--nuweight', type=float, default=1e3,
            help="nu constraint weight")
        # fit options 
        parser.add_argument("--unweighted", action="store_true",
            help="perform unweighted least squares fit")
        parser.add_argument("--sklearn", action="store_true",
            help="use sklearn linear regression instead of scipy lstsq")
        # scipy fit options
        parser.add_argument("--max-iter", type=int, default=100,
            help="max number of iterations to use in lsqr")
        parser.add_argument("--atol", type=float, default=1e-4,
            help="a stopping tolerance")
        parser.add_argument("--btol", type=float, default=1e-8,
            help="b stopping tolerance")