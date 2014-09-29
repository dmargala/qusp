import inspect
import numpy as np
import scipy.sparse
import h5py

import bosslya

class ContinuumModel(object):
    def __init__(self, obsmin, obsmax, restmin, restmax, nrestbins, tiltwave,
        absmin, absmax, absmodelexp, verbose=False):
        self.verbose = verbose

        # initialize binning arrays
        assert obsmax > obsmin, ('obsmax must be greater than obsmin')
        self.obsWaveMin = obsmin
        self.obsWaveMax = obsmax

        obsFiducialWave = bosslya.wavelength.getFiducialWavelength(np.arange(4800))
        self.obsWaveMinIndex = np.argmax(obsFiducialWave > obsmin)
        self.obsWaveMaxIndex = np.argmax(obsFiducialWave > obsmax)+1

        self.obsWaveCenters = bosslya.wavelength.getFiducialWavelength(np.arange(self.obsWaveMinIndex,self.obsWaveMaxIndex))
        self.obsNParams = len(self.obsWaveCenters)

        if verbose:
            print 'Observed frame bin centers span [%.2f:%.2f] with %d bins.' % (
                self.obsWaveCenters[0],self.obsWaveCenters[-1],self.obsNParams)

        assert restmax > restmin, ('restmin must be greater than restmax')
        self.restWaveMin = restmin
        self.restWaveMax = restmax
        self.restNParams = nrestbins
        self.restWaveDelta = float(restmax-restmin)/nrestbins
        self.restWaveCenters = 0.5*self.restWaveDelta + np.linspace(
            restmin,restmax,nrestbins,endpoint=False)

        if verbose:
            print 'Rest frame bin centers span [%.2f:%.2f] with %d bins.' % (
                self.restWaveCenters[0],self.restWaveCenters[-1],self.restNParams)

        self.absmodelexp = absmodelexp
        self.absMin = max(absmin,restmin)
        self.absMax = min(absmax,restmax)
        self.absMinIndex = np.argmax(self.restWaveCenters >= self.absMin)
        self.absMaxIndex = np.argmax(self.restWaveCenters > self.absMax)
        self.absWaveCenters = self.restWaveCenters[self.absMinIndex:self.absMaxIndex]
        self.absNParams = len(self.absWaveCenters)

        if verbose:
            if self.absNParams > 0:
                print 'Absorption bin centers span [%.2f:%.2f] with %d bins.' % (
                    self.absWaveCenters[0], self.absWaveCenters[-1], self.absNParams)
            else:
                print 'No absorption params'

        self.targetNParams = 1

        self.tiltwave = tiltwave
        if tiltwave > 0:
            self.targetNParams += 1
        # the number of "model" pixels (excluding per target params)
        self.nModelPixels = self.obsNParams + self.restNParams + self.absNParams
        # sparse matrix entry holders
        self.rowIndices = list()
        self.colIndices = list()
        self.coefficients = list()
        self.yvalues = list()

        self.model = None
        self.y = None

        self.amplitude = list()
        self.nu = list()

        self.nTotalPixels = 0
        self.nTargets = 0
        self.nconstraints = 0

    def addObservation(self, target, flux, wave, ivar, unweighted=True):
        """
        Adds an observation to be fit. Returns the number of pixels added. The provided
        target argument must have an attribute named 'z' with the target's redshift.

        weighted fit not yet implemented.

        """
        assert unweighted, ('Weighted fit not implemented yet...')

        try:
            z = target.z
        except AttributeError:
            print 'Target does not have z attribute.'
            raise

        # this spectrum's wavelength axis pixel offset
        fiducialOffset = bosslya.wavelength.getFiducialPixelIndexOffset(np.log10(wave[0]))

        obsFiducialIndices = fiducialOffset+np.arange(len(wave))
        obsFiducialWave = bosslya.wavelength.getFiducialWavelength(obsFiducialIndices)

        restWave = obsFiducialWave/(1+target.z)
        restIndices = np.floor((restWave - self.restWaveMin)/self.restWaveDelta).astype(int)

        # trim ranges to valid data
        validbins = np.all((
            obsFiducialIndices >= self.obsWaveMinIndex, 
            obsFiducialIndices  < self.obsWaveMaxIndex, 
            restIndices < self.restNParams, 
            restIndices >= 0, 
            flux > 0, ivar > 0), axis=0)

        nPixels = np.sum(validbins)
        if nPixels <= 0:
            if self.verbose:
                print 'No good pixels in relavant range on target %s (z=%.2f)' % (target, target.z)
            return 0

        try:
            amp = target.amp
        except AttributeError:
            amp = None
        self.amplitude.append(amp)

        try:
            nu = target.nu
        except AttributeError:
            nu = None
        self.nu.append(nu)

        yvalues = np.log(flux[validbins]) + np.log(1+target.z)

        '''
        # compute weights
        if unweighted:
            weights = np.ones(nPixels)
        else:
            weights = ivar[validbins]
        sqrtw = np.sqrt(weights)

        # Append yvalu values
        yvalues = sqrtw*yvalues
        '''

        # Assemble matrix
        colIndices = []
        rowIndices = []
        coefficients = []
        matrixOffset = {'row': self.nTotalPixels, 'col': 0}

        # helper function to assemble seperate blocks at a time
        def assembleBlock(rows, cols, paramValues, nparams):
            # Each col corresponds to model parameter value, the model matrix
            # is ordered in blocks of model parameters
            colIndices.append(matrixOffset['col'] + cols)
            # Each row corresponds to single flux value, the model matrix
            # will have nParams entries per row
            rowIndices.append(matrixOffset['row'] + rows)
            # The coefficients in the model matrix are the sqrt(weight), unless a 'coef'
            # function is specified in the param dictionary
            coefficients.append(paramValues)
            matrixOffset['col'] += nparams

        # build obs model param block
        obsIndices = obsFiducialIndices[validbins]-self.obsWaveMinIndex
        assert np.amax(obsIndices) < self.obsNParams, (
            'Invalid obsmodel index value')

        assembleBlock(np.arange(nPixels), obsIndices, np.ones(nPixels), self.obsNParams)

        # build rest model param block
        restIndices = restIndices[validbins]
        assert np.amax(restIndices) < self.restNParams, (
            'Invalid rest model index value')

        assembleBlock(np.arange(nPixels), restIndices, np.ones(nPixels), self.restNParams)

        # build absorption model param block
        absMinIndex = np.argmax(restIndices == self.absMinIndex)
        absMaxIndex = np.argmax(restIndices == self.absMaxIndex)

        if absMaxIndex > absMinIndex:
            absRows = np.arange(nPixels)[absMinIndex:absMaxIndex]
            absIndices = restIndices[absMinIndex:absMaxIndex] - self.absMinIndex

            assert np.amax(absIndices) < self.absNParams, 'Invalid abs index value'
            absValues = -np.ones(len(absIndices))*np.power(1+target.z,self.absmodelexp)

            assembleBlock(absRows, absIndices, absValues, self.absNParams)

        # build target param block
        targetIndices = self.targetNParams*self.nTargets*np.ones(nPixels)
        # amplitude param
        assembleBlock(np.arange(nPixels), targetIndices, np.ones(nPixels), 1)
        # spectral tilt index param
        if self.tiltwave > 0:
            assembleBlock(np.arange(nPixels), targetIndices, np.log(self.restWaveCenters[restIndices]/self.tiltwave), 1)

        # add to assembled blocks for this observation to the model
        self.addModelCoefficents(np.concatenate(rowIndices),
            np.concatenate(colIndices), np.concatenate(coefficients), yvalues)
        self.nTargets += 1

        return nPixels

    def addRestConstraint(self, yvalue, wavemin, wavemax, weight):
        waves = self.restWaveCenters
        offset = self.obsNParams

        waveIndexRange = np.arange(np.argmax(waves > wavemin), np.argmax(waves > wavemax))
        constraintCoefficients = weight*np.ones(len(waveIndexRange))

        if self.verbose:
            print 'Adding constraint: sum(%.2g*logC([%.2f:%.2f])) = %.1f (%d logC params [%d:%d])' % (
                weight, waves[waveIndexRange[0]], waves[waveIndexRange[-1]], yvalue, 
                len(waveIndexRange), waveIndexRange[0], waveIndexRange[-1])

        colIndices = offset+waveIndexRange
        rowIndices = self.nTotalPixels*np.ones(len(constraintCoefficients))

        self.addModelCoefficents(rowIndices, colIndices, constraintCoefficients, [yvalue])
        self.nconstraints += 1

    def addObsConstraint(self, yvalue, wavemin, wavemax, weight):
        waves = self.obsWaveCenters
        offset = 0

        waveIndexRange = np.arange(np.argmax(waves > wavemin), np.argmax(waves > wavemax)+1)
        nconstraints = len(waveIndexRange)

        constraintCoefficients = weight*np.ones(nconstraints)

        if self.verbose:
            print 'Adding constraint: %.2g*logT([%.2f:%.2f]) = %.1f (%d logT params [%d:%d])' % (
                weight, waves[waveIndexRange[0]], waves[waveIndexRange[-1]], yvalue, 
                len(waveIndexRange), waveIndexRange[0], waveIndexRange[-1])

        colIndices = offset+waveIndexRange
        rowIndices = self.nTotalPixels+np.arange(nconstraints)

        yvalues = yvalue*np.ones(nconstraints)

        self.addModelCoefficents(rowIndices, colIndices, constraintCoefficients, yvalues)
        self.nconstraints += nconstraints

    def addTiltConstraint(self, weight):
        
        colIndices = 1 + self.nModelPixels + np.arange(0,self.targetNParams*self.nTargets,self.targetNParams)

        assert len(colIndices) == self.nTargets, ('Invalid number of nu params')

        if self.verbose:
            print 'Adding constraint: %.2g*sum(nu) = 0 (%d nu params)' % (weight,self.nTargets)

        rowIndices = self.nTotalPixels*np.ones(self.nTargets)
        constraintCoefficients = weight*np.ones(self.nTargets)/self.nTargets

        self.addModelCoefficents(rowIndices, colIndices, constraintCoefficients, [0])
        self.nconstraints += 1

    def addModelCoefficents(self, rows, cols, coefs, yvalues):
        self.colIndices.append(cols)
        self.rowIndices.append(rows)
        self.coefficients.append(coefs)
        self.yvalues.append(yvalues)
        self.nTotalPixels += len(yvalues)

    def finalize(self):
        """
        Does final assembly of the sparse matrix representing the model.
        """
        nModelPixels = self.nModelPixels + self.targetNParams*self.nTargets

        rowIndices = np.concatenate(self.rowIndices)
        colIndices = np.concatenate(self.colIndices)
        coefficients = np.concatenate(self.coefficients)
        yvalues = np.concatenate(self.yvalues)

        self.y = yvalues

        # build the sparse matrix
        model = scipy.sparse.coo_matrix((coefficients,(rowIndices,colIndices)), 
            shape=(self.nTotalPixels,nModelPixels), dtype=np.float32)
        # convert the sparse matrix to compressed sparse column format
        self.model = model.tocsc()

        if self.verbose:
            print 'Number of transmission model params: %d' % self.obsNParams
            print 'Number of continuum model params: %d' % self.restNParams
            print 'Number of absorption model params: %d' % self.absNParams
            print 'Number of targets: %d' % self.nTargets
            print 'Number of target params: %d' % self.targetNParams
            print ''
            print 'Total number of model params: %d' % nModelPixels
            print 'Total number of flux measurements: %d (%d constraints)' % (self.nTotalPixels, self.nconstraints)
            print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)\n' % (
                self.model.data.nbytes, self.model.indptr.nbytes, self.model.indices.nbytes)

    def getModel(self):
        if self.model is None:
            self.finalize()
        return self.model, self.y

    def getResults(self, soln):
        """
        Returns a dictionary containing fit results
        """
        # assert self.soln is not None, ('Can\'t request results before fitting')
        results = dict()
        offset = 0

        # transform logT -> T
        results['transmission'] = np.exp(soln[offset:offset+self.obsNParams])
        offset += self.obsNParams

        # transform logC -> C
        results['continuum'] = np.exp(soln[offset:offset+self.restNParams])
        offset += self.restNParams

        # absorption
        results['absorption'] = soln[offset:offset+self.absNParams]
        offset += self.absNParams

        # transform logA -> A
        results['amplitude'] = np.exp(soln[offset:offset+self.targetNParams*self.nTargets:self.targetNParams])
        offset += 1

        # spectral tilt
        if self.tiltwave > 0:
            results['nu'] = soln[offset:offset+self.targetNParams*self.nTargets:self.targetNParams]
            offset += 1

        # return dictionary of parameters
        return results

    def getChiSq(self, soln):
        """
        Returns chisq of best fit
        """
        assert self.model is not None, ('Can\'t request chisq before model assembly.')
        # calculate residuals
        residuals = self.y - self.model.dot(soln)
        # return chisq
        return np.dot(residuals,residuals)/len(residuals)

    def getObservationChiSq(self, soln, i):
        """
        Returns chisq of the specified observation index
        """
        assert self.model is not None, ('Can\'t request chisq before model assembly.')
        # check soln size
        nModelPixels = self.nModelPixels + self.targetNParams*self.nTargets
        assert len(soln) == nModelPixels, ('Size of soln does not match model.')
        # pick out y value entries for the requested observation
        yvalues = self.yvalues[i]
        nPixels = len(yvalues)
        # pick out model rows corresponding to the requested observation
        rowIndices = self.rowIndices[i] - min(self.rowIndices[i])
        colIndices = self.colIndices[i]
        coefficients = self.coefficients[i]
        # build the sparse matrix
        model = scipy.sparse.coo_matrix((coefficients,(rowIndices,colIndices)), 
            shape=(nPixels,nModelPixels), dtype=np.float32)
        # convert the sparse matrix to compressed sparse column format
        model = model.tocsc()
        '''
        rowIndices = self.rowIndices[i]
        model = self.model[min(rowIndices),min(rowIndices)]
        '''
        # calculate residuals
        residuals = yvalues - model.dot(soln)
        # return chisq
        return np.dot(residuals, residuals)/len(residuals)

    def save(self, filename, soln, args):
        """
        Writes the model, solution, and results to hdf5 file to provided filename.
        """
        # open hdf5 output file
        outfile = h5py.File(filename,'w')

        outfile.create_dataset('model_data', data=self.model.data)
        outfile.create_dataset('model_indices', data=self.model.indices)
        outfile.create_dataset('model_indptr', data=self.model.indptr)
        outfile.create_dataset('model_shape', data=self.model.shape)
        outfile.create_dataset('soln', data=soln)

        dsetObsWave = outfile.create_dataset('obsWaveCenters', data=self.obsWaveCenters)
        dsetRestWave = outfile.create_dataset('restWaveCenters', data=self.restWaveCenters)

        results = self.getResults(soln)
        obsModelValues = results['transmission']
        restModelValues = results['continuum']
        targetModelValues = results['amplitude']
        absModelValues = results['absorption']
        tiltModelValues = results['nu']

        dsetT = outfile.create_dataset('transmission', data=obsModelValues)
        dsetT.attrs['normmin'] = args.obsnormmin
        dsetT.attrs['normmax'] = args.obsnormmax
        dsetT.attrs['normweight'] = args.obsnormweight

        dsetC = outfile.create_dataset('continuum', data=restModelValues)
        dsetC.attrs['normmin'] = args.restnormmin
        dsetC.attrs['normmax'] = args.restnormmax
        dsetC.attrs['normweight'] = args.restnormweight

        dsetA = outfile.create_dataset('amplitude', data=targetModelValues)

        dsetabs = outfile.create_dataset('absorption', data=absModelValues)
        dsetabs.attrs['minRestIndex'] = self.absMinIndex
        dsetabs.attrs['maxRestIndex'] = self.absMaxIndex 
        dsetabs.attrs['absmodelexp'] = self.absmodelexp

        dsetTilt = outfile.create_dataset('nu', data=tiltModelValues)
        dsetTilt.attrs['tiltwave'] = args.tiltwave
        dsetTilt.attrs['tiltweight'] = args.tiltweight
    
        chiSqs = [self.getObservationChiSq(soln, i) for i in range(self.nTargets)]
        outfile.create_dataset('chisq', data=chiSqs)

        # return h5py file
        return outfile

    @staticmethod
    def addArgs(parser):
        """
        Add arguments to the provided command-line parser that support the fromArgs() method.
        """
        # transmission model wavelength grid options
        parser.add_argument("--obsmin", type=float, default=3600,
            help="transmission model wavelength minimum")
        parser.add_argument("--obsmax", type=float, default=10000,
            help="transmission model wavelength maximum")
        parser.add_argument("--obsnormmin", type=float, default=3600,
            help="obsframe wavelength to normalize at")
        parser.add_argument("--obsnormmax", type=float, default=10000,
            help="obsframe window size +/- on each side of obsnorm wavelength")
        parser.add_argument("--obsnormweight", type=float, default=1e1,
            help="norm constraint weight")
        # continuum model wavelength grid options
        parser.add_argument("--restmin", type=float, default=900,
            help="rest wavelength minimum")
        parser.add_argument("--restmax", type=float, default=2900,
            help="rest wavelength maximum")
        parser.add_argument("--nrestbins", type=int, default=1000,
            help="number of restframe bins")
        parser.add_argument("--restnormmin", type=float, default=1275,
            help="restframe wavelength to normalize at")
        parser.add_argument("--restnormmax", type=float, default=1285,
            help="restframe window size +/- on each side of restnorm wavelength")
        parser.add_argument("--restnormweight", type=float, default=1e3,
            help="norm constraint weight")
        # absorption model parameter options
        parser.add_argument("--absmin", type=float,default=900,
            help="absorption min wavelength (rest frame)")
        parser.add_argument("--absmax", type=float,default=1400,
            help="absoprtion max wavelength (rest frame)")
        parser.add_argument("--absmodelexp", type=float, default=3.92,
            help="absorption model (1+z) factor exponent")
        # spectral tilt parameter options
        parser.add_argument("--tiltwave", type=float, default=1280,
            help="spectral tilt pivot wavelength")
        parser.add_argument("--tiltweight", type=float, default=1e3,
            help="spectral tilt constraint weight")
        # fit options 
        parser.add_argument("--unweighted", action="store_true",
            help="perform unweighted least squares fit")

    @staticmethod
    def fromArgs(args):
        """
        Returns a dictionary of constructor parameter values based on the parsed args provided.
        """
        # Look up the named ContinuumModel constructor parameters.
        pnames = (inspect.getargspec(ContinuumModel.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        argsDict = vars(args)
        # Return a dictionary of constructor parameters provided in args.
        return { key:argsDict[key] for key in (set(pnames) & set(argsDict)) }
