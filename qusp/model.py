import inspect
import numpy as np
import scipy.sparse
import h5py

import qusp

class ContinuumModel(object):
    """
    Represents a linearized quasar continuum model.
    """
    def __init__(self, obsmin, obsmax, restmin, restmax, nrestbins, tiltwave,
        absmin, absmax, absmodelexp, absscale, verbose=False):
        """
        Initializes a linearized quasar continuum model using the specified
        parameter limits and values.
        """
        self.verbose = verbose
        # initialize transmission model params
        assert obsmax > obsmin, ('obsmax must be greater than obsmin')
        self.obsWaveMin = obsmin
        self.obsWaveMax = obsmax
        obsFiducialWave = qusp.wavelength.getFiducialWavelength(np.arange(4800))
        self.obsWaveMinIndex = np.argmax(obsFiducialWave > obsmin)
        self.obsWaveMaxIndex = np.argmax(obsFiducialWave > obsmax)+1
        self.obsWaveCenters = qusp.wavelength.getFiducialWavelength(np.arange(self.obsWaveMinIndex,self.obsWaveMaxIndex))
        self.obsNParams = len(self.obsWaveCenters)
        if verbose:
            print 'Observed frame bin centers span [%.2f:%.2f] with %d bins.' % (
                self.obsWaveCenters[0],self.obsWaveCenters[-1],self.obsNParams)
        # initialize continuum model params
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
        # initialize absorption model params
        self.absmodelexp = absmodelexp
        self.absMin = max(absmin,restmin)
        self.absMax = min(absmax,restmax)
        self.absMinIndex = np.argmax(self.restWaveCenters >= self.absMin)
        self.absMaxIndex = np.argmax(self.restWaveCenters > self.absMax)
        self.absWaveCenters = self.restWaveCenters[self.absMinIndex:self.absMaxIndex]
        self.absNParams = len(self.absWaveCenters)
        self.absScale = absscale
        if verbose:
            if self.absNParams > 0:
                print 'Absorption bin centers span [%.2f:%.2f] with %d bins.' % (
                    self.absWaveCenters[0], self.absWaveCenters[-1], self.absNParams)
            else:
                print 'No absorption params'
        # spectral tilt pivot wavelength
        self.tiltwave = tiltwave
        # the number of model params (excluding per target params)
        self.nModelParams = self.obsNParams + self.restNParams + self.absNParams
        if verbose:
            print 'Fit model initialized with %d model params.\n' % self.nModelParams
        # sparse matrix entry holders
        self.yvalues = list()
        self.obsBlocks = list()
        self.constraintBlocks = list()
        self.amplitude = list()
        self.normCoefficients = list()
        self.nu = list()
        self.tiltCoefficients = list()
        # store finalized model pieces 
        self.model = None
        self.y = None
        # model stats
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
            z = target['z']
        except KeyError:
            print 'Target does not have z attribute.'
            raise
        # this spectrum's co-add wavelength axis pixel offset
        fiducialOffset = qusp.wavelength.getFiducialPixelIndexOffset(np.log10(wave[0]))
        # map pixels to observed frame wavelength grid
        obsFiducialIndices = fiducialOffset+np.arange(len(wave))
        obsFiducialWave = qusp.wavelength.getFiducialWavelength(obsFiducialIndices)
        # map pixels to rest frame wavelength grid
        restWave = obsFiducialWave/(1+z)
        restIndices = np.floor((restWave - self.restWaveMin)/self.restWaveDelta).astype(int)
        # trim ranges to valid data
        validbins = np.all((
            obsFiducialIndices >= self.obsWaveMinIndex, 
            obsFiducialIndices  < self.obsWaveMaxIndex, 
            restIndices < self.restNParams, 
            restIndices >= 0, 
            flux > 0, ivar > 0), axis=0)
        nPixels = np.sum(validbins)
        # skip target if no valid pixels
        if nPixels <= 0:
            if self.verbose:
                print 'No good pixels in relevant range on target %s (z=%.2f)' % (target['target'], z)
            return 0
        # initialize y values to logf + log(1+z)
        yvalues = np.log(flux[validbins]) + np.log(1+z)
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
        obsBlocks = []
        # build transmission model param block
        obsIndices = obsFiducialIndices[validbins]-self.obsWaveMinIndex
        assert np.amax(obsIndices) < self.obsNParams, ('Invalid obsmodel index value')
        transBlock = scipy.sparse.coo_matrix((np.ones(nPixels),(np.arange(nPixels),obsIndices)), 
            shape=(nPixels,self.obsNParams))
        obsBlocks.append(transBlock)
        # build continuum model param block
        restIndices = restIndices[validbins]
        assert np.amax(restIndices) < self.restNParams, ('Invalid rest model index value')
        contBlock = scipy.sparse.coo_matrix((np.ones(nPixels),(np.arange(nPixels),restIndices)), 
            shape=(nPixels,self.restNParams))
        obsBlocks.append(contBlock)
        # build absorption model param block
        absMinIndex = np.argmax(restIndices == self.absMinIndex)
        absMaxIndex = np.argmax(restIndices == self.absMaxIndex)
        # check if any of the this observation has pixels in the relevant absorption range
        if absMaxIndex > absMinIndex:
            absRows = np.arange(nPixels)[absMinIndex:absMaxIndex]
            absCols = restIndices[absMinIndex:absMaxIndex] - self.absMinIndex
            assert np.amax(absCols) < self.absNParams, 'Invalid abs param index value'
            absCoefs = -np.ones(len(absCols))*np.power(1+z,self.absmodelexp)/self.absScale
            absBlock = scipy.sparse.coo_matrix((absCoefs,(absRows,absCols)), 
                shape=(nPixels,self.absNParams))
            obsBlocks.append(absBlock)
        elif self.absNParams > 0:
            absBlock = scipy.sparse.coo_matrix((nPixels,self.absNParams))
            obsBlocks.append(absBlock)
        else:
            # no absorption parameters, do nothing
            pass
        # process amplitude params
        try:
            amp = target['amp']
            yvalues -= np.log(amp)
            self.normCoefficients.append(None)
        except KeyError:
            amp = None
            self.normCoefficients.append(np.ones(nPixels))
            self.nModelParams += 1
        self.amplitude.append(amp)
        # process spectral tilt params
        tiltCoefficients = np.log(self.restWaveCenters[restIndices]/self.tiltwave)
        try:
            nu = target['nu']
            yvalues -= nu*tiltCoefficients
            self.tiltCoefficients.append(None)
        except KeyError:
            nu = None
            self.tiltCoefficients.append(tiltCoefficients)
            self.nModelParams += 1
        self.nu.append(nu)
        # append this observation's blocks to the model
        self.obsBlocks.append(obsBlocks)
        # append this observation's y values to the model
        self.yvalues.append(yvalues)
        # update model stats
        self.nTotalPixels += nPixels
        self.nTargets += 1
        # return number of pixels added from this observation
        return nPixels

    def addObsConstraint(self, yvalue, wavemin, wavemax, weight):
        """
        Adds a constraint equation for each of the transmission model params between
        the specified observed frame wavelengths.
        """
        # transmission model params are first
        offset = 0
        # find range of transmission model params in constraint window
        waves = self.obsWaveCenters
        waveIndexRange = np.arange(np.argmax(waves > wavemin), np.argmax(waves > wavemax)+1)
        # number of transmission parameters in constraint window
        nTransmissionParams = len(waveIndexRange)
        # build constraint block
        coefs = weight*np.ones(nTransmissionParams)
        rows = np.arange(nTransmissionParams)
        cols = offset+waveIndexRange
        block = scipy.sparse.coo_matrix((coefs,(rows,cols)),shape=(nTransmissionParams,self.nModelParams))
        if self.verbose:
            print 'Adding constraint: %.2g*logT([%.2f:%.2f]) = %.1f (%d logT params [%d:%d])' % (
                weight, waves[waveIndexRange[0]], waves[waveIndexRange[-1]], yvalue, 
                nTransmissionParams, waveIndexRange[0], waveIndexRange[-1])
        # append constraint block and update number of constraint equations
        self.constraintBlocks.append(block)
        self.nconstraints += nTransmissionParams
        # append yvalues and update total number of rows
        yvalues = yvalue*np.ones(nTransmissionParams)
        self.yvalues.append(yvalues)
        self.nTotalPixels += nTransmissionParams

    def addRestConstraint(self, yvalue, wavemin, wavemax, weight):
        """
        Adds a constraint equation on the geometric mean of the continuum model in
        between the specified rest frame wavelengths.
        """
        # rest model params are after obs model params
        offset = self.obsNParams
        # find range of continuum model params in constraint window
        waves = self.restWaveCenters
        waveIndexRange = np.arange(np.argmax(waves > wavemin), np.argmax(waves > wavemax))
        # number of continuum parameters in constraint window
        nContinuumParams = len(waveIndexRange)
        # build constraint block
        coefs = weight*np.ones(nContinuumParams)
        rows = np.zeros(nContinuumParams)
        cols = offset+waveIndexRange
        block = scipy.sparse.coo_matrix((coefs,(rows,cols)),shape=(1,self.nModelParams))
        if self.verbose:
            print 'Adding constraint: sum(%.2g*logC([%.2f:%.2f])) = %.1f (%d logC params [%d:%d])' % (
                weight, waves[waveIndexRange[0]], waves[waveIndexRange[-1]], yvalue, 
                nContinuumParams, waveIndexRange[0], waveIndexRange[-1])
        # append constraint block and update number of constraint equations
        self.constraintBlocks.append(block)
        self.nconstraints += 1
        # append yvalues and update total number of rows
        self.yvalues.append([yvalue])
        self.nTotalPixels += 1

    def addTiltConstraint(self, weight):
        """
        Adds a constraint equation on the mean of the non-fixed spectral tilt params.
        """
        # calculate tilt block column offset, tilt params are after normalization params
        offset = self.obsNParams + self.restNParams + self.absNParams + self.amplitude.count(None)
        # count number of free tilt params
        nTiltParams = self.nu.count(None)
        # build constraint block
        coefs = weight*np.ones(nTiltParams)/nTiltParams
        rows = np.zeros(nTiltParams)
        cols = offset+np.arange(nTiltParams)
        block = scipy.sparse.coo_matrix((coefs,(rows,cols)),shape=(1,self.nModelParams))
        if self.verbose:
            print 'Adding constraint: %.2g*sum(nu) = 0 (%d nu params)' % (weight,nTiltParams)
        # append constraint block and update number of constraint equations
        self.constraintBlocks.append(block)
        self.nconstraints += 1
        # append yvalues and update total number of rows
        self.yvalues.append([0])
        self.nTotalPixels += 1

    def finalize(self):
        """
        Does final assembly of the sparse matrix representing the model.
        """
        # pass through each observation and do final assembly of target param blocks
        for i in range(self.nTargets):
            nPixels = self.obsBlocks[i][0].shape[0]
            # add norm block
            if self.amplitude.count(None) > 0:
                a = self.amplitude[i]
                if a is None:
                    ampIndices = self.amplitude[:i].count(None)*np.ones(nPixels)
                    normBlock = scipy.sparse.coo_matrix((self.normCoefficients[i],(np.arange(nPixels),ampIndices)),
                        shape=(nPixels,self.amplitude.count(None)))
                else:
                    normBlock = scipy.sparse.coo_matrix((nPixels,self.amplitude.count(None)))
                self.obsBlocks[i].append(normBlock)
            # add tilt block
            if self.nu.count(None) > 0:
                nu = self.nu[i]
                if nu is None:
                    nuIndices = self.nu[:i].count(None)*np.ones(nPixels)
                    tiltBlock = scipy.sparse.coo_matrix((self.tiltCoefficients[i],(np.arange(nPixels),nuIndices)),
                        shape=(nPixels,self.nu.count(None)))
                else:
                    tiltBlock = scipy.sparse.coo_matrix((nPixels,self.nu.count(None)))
                self.obsBlocks[i].append(tiltBlock)
        # combine blocks from all observations
        obsBlock = scipy.sparse.bmat(self.obsBlocks)
        # combine blocks from all constraints
        contraintBlock = scipy.sparse.vstack(self.constraintBlocks)
        # comebine observations and constraints
        finalModel = scipy.sparse.vstack([obsBlock,contraintBlock])

        assert finalModel.shape[0] == self.nTotalPixels
        assert finalModel.shape[1] == self.nModelParams

        self.model = finalModel.tocsc()

        # concatenate y values
        yvalues = np.concatenate(self.yvalues)
        self.y = yvalues

        if self.verbose:
            print 'Number of transmission model params: %d' % self.obsNParams
            print 'Number of continuum model params: %d' % self.restNParams
            print 'Number of absorption model params: %d' % self.absNParams
            print 'Number of targets: %d' % self.nTargets
            print 'Number of amplitude params: %d' % self.amplitude.count(None)
            print 'Number of tilt params: %d' % self.nu.count(None)
            print ''
            print 'Total number of model params: %d' % self.nModelParams
            print 'Total number of flux measurements: %d (%d constraints)' % (self.nTotalPixels, self.nconstraints)
            print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)\n' % (
                self.model.data.nbytes, self.model.indptr.nbytes, self.model.indices.nbytes)

    def getModel(self):
        """
        Returns the assembled model matrix and corresponding y values
        """
        if self.model is None:
            self.finalize()
        return self.model, self.y

    def getResults(self, soln):
        """
        Returns a dictionary containing fit results
        """
        assert self.model is not None, ('Can\'t request results before model assembly.')
        assert len(soln) == self.nModelParams, ('Size of soln does not match model.')
        results = dict()
        offset = 0
        # transmission: transform logT -> T
        results['transmission'] = np.exp(soln[offset:offset+self.obsNParams])
        offset += self.obsNParams
        # continuum: transform logC -> C
        results['continuum'] = np.exp(soln[offset:offset+self.restNParams])
        offset += self.restNParams
        # absorption
        results['absorption'] = soln[offset:offset+self.absNParams]*self.absScale
        offset += self.absNParams
        # amplitude: transform logA -> A
        amplitude = list()
        ampindex = 0
        for amp in self.amplitude:
            if amp is None:
                amp = np.exp(soln[offset+ampindex])
                ampindex += 1
            amplitude.append(amp)
        results['amplitude'] = amplitude
        offset += self.amplitude.count(None)
        # spectral tilt
        nu = list()
        nuindex = 0
        for nuvalue in self.nu:
            if nuvalue is None:
                nuvalue = soln[offset+nuindex]
                nuindex += 1
            nu.append(nuvalue)
        results['nu'] = nu
        offset += self.nu.count(None)
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
        assert len(soln) == self.nModelParams, ('Size of soln does not match model.')
        # pick out y value entries for the requested observation
        yvalues = self.yvalues[i]
        # pick out model rows for the requested observation
        model = scipy.sparse.hstack(self.obsBlocks[i]).tocsc()
        # calculate residuals
        residuals = yvalues - model.dot(soln)
        # return chisq
        return np.dot(residuals, residuals)/len(residuals)

    def save(self, filename, soln, args, saveModel=True):
        """
        Saves soln to the specified filename as an hdf5 file. Parsed results and fit meta data
        are also saved. Use the saveModel arg to indicate whether or not to save the raw
        data of the sparse matrix model.
        """
        results = self.getResults(soln)
        # open hdf5 output file
        outfile = h5py.File(filename,'w')
        # save soln
        outfile.create_dataset('soln', data=soln)
        if saveModel:
            # save model data
            outfile.create_dataset('model_data', data=self.model.data)
            outfile.create_dataset('model_indices', data=self.model.indices)
            outfile.create_dataset('model_indptr', data=self.model.indptr)
            outfile.create_dataset('model_shape', data=self.model.shape)
            outfile.create_dataset('y', data=self.y)
        # save wavelength grids
        dsetObsWave = outfile.create_dataset('obsWaveCenters', data=self.obsWaveCenters)
        dsetRestWave = outfile.create_dataset('restWaveCenters', data=self.restWaveCenters)
        # save transmission model params and relevant info
        dsetT = outfile.create_dataset('transmission', data=results['transmission'])
        dsetT.attrs['normmin'] = args.obsnormmin
        dsetT.attrs['normmax'] = args.obsnormmax
        dsetT.attrs['normweight'] = args.obsnormweight
        # save continuum model params and relevant info
        dsetC = outfile.create_dataset('continuum', data=results['continuum'])
        dsetC.attrs['normmin'] = args.restnormmin
        dsetC.attrs['normmax'] = args.restnormmax
        dsetC.attrs['normweight'] = args.restnormweight
        # save absorption model params and relevant info
        dsetabs = outfile.create_dataset('absorption', data=results['absorption'])
        dsetabs.attrs['minRestIndex'] = self.absMinIndex
        dsetabs.attrs['maxRestIndex'] = self.absMaxIndex 
        dsetabs.attrs['absmodelexp'] = self.absmodelexp
        # save amplitude params and relevant info
        dsetA = outfile.create_dataset('amplitude', data=results['amplitude'])
        # save spectral tilt params and relevant info
        dsetTilt = outfile.create_dataset('nu', data=results['nu'])
        dsetTilt.attrs['tiltwave'] = args.tiltwave
        dsetTilt.attrs['tiltweight'] = args.tiltweight
        # save per-obs chisqs
        chiSqs = [self.getObservationChiSq(soln, i) for i in range(self.nTargets)]
        outfile.create_dataset('chisq', data=chiSqs)
        # return h5py output file
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
        parser.add_argument("--absmin", type=float, default=900,
            help="absorption min wavelength (rest frame)")
        parser.add_argument("--absmax", type=float, default=1400,
            help="absoprtion max wavelength (rest frame)")
        parser.add_argument("--absmodelexp", type=float, default=3.92,
            help="absorption model (1+z) factor exponent")
        parser.add_argument("--absscale", type=float, default=1,
            help="scale absorption params in fit")
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
