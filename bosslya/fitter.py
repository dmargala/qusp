import numpy as np

import scipy.sparse
import scipy.sparse.linalg

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
        self.soln = None
        self.nconstraints = 0

    def addObservation(self, logFlux, obsSlice, restSlice, weights=None):
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
        if weights is None:
            sqrtw = np.ones(nPixels)
        else:
            assert len(weights) == nPixels, (
                'Weights array size does not match data')
            sqrtw = np.sqrt(weights)
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
                colIndices.append(colOffset+obsSlice)
                colOffset += self.nObs
            elif param['type'] is 'rest':
                colIndices.append(colOffset+restSlice)
                colOffset += self.nRest
            elif param['type'] is 'target':
                colIndices.append(colOffset+self.nTargets*np.ones(nPixels))
                colOffset += 1
        self.colIndices.append(np.concatenate(colIndices))

        # The coefficients in the model matrix are the weights, unless a 'coef'
        # function is specified in the param dictionary
        for i,param in enumerate(self.params):
            if 'coef' in param.keys():
                coefficients.append(sqrtw*param['coef'](self.obsWaveCenters[obsSlice], self.restWaveCenters[restSlice]))
            else:
                coefficients.append(sqrtw)
        self.coefficients.append(np.concatenate(coefficients))

        # Increment the total number of pixel values and the number of observations
        self.nTotalPixels += nPixels
        self.nTargets += 1

    def addConstraint(self, paramName, logFlux, index, coef):

        assert len(index) == len(coef), ('index and coef must be same length')  

        colOffset = 0
        for i,param in enumerate(self.params):
            if param['name'] is paramName:
                self.colIndices.append(colOffset+index)
            if param['type'] is 'obs':
                colOffset += self.nObs
            elif param['type'] is 'rest':
                colOffset += self.nRest
            elif param['type'] is 'target':
                colOffset += 1

        self.coefficients.append(coef)
        self.rowIndices.append(self.nTotalPixels*np.ones(len(index)))
        self.logFluxes.append([logFlux])

        self.nTotalPixels += 1
        self.nconstraints += 1        

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
        self.model = model.tocsc()

        if verbose:
            print 'Number of transmission model params: %d' % self.nObs
            print 'Number of continuum model params: %d' % self.nRest
            print 'Number of targets: %d' % self.nTargets
            print 'Total model params: %d' % nModelPixels
            print 'Total number of flux measurements: %d' % self.nTotalPixels
            print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)' % (
                self.model.data.nbytes, self.model.indptr.nbytes, self.model.indices.nbytes)

        # perform fit
        if sklearn:
            from sklearn import linear_model
            regr = linear_model.LinearRegression()
            if verbose:
                print '... performing fit using sklearn.linear_model.LinearRegression ... '
            regr.fit(self.model, logFluxes)
            self.soln = regr.coef_
        else:
            if verbose:
                print '... sperforming fit using cipy.sparse.linalg.lsqr ... '
            soln = scipy.sparse.linalg.lsqr(self.model, logFluxes, show=verbose,
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
                npixels = self.nObs
            elif param['type'] is 'rest':
                npixels = self.nRest
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
        return np.dot(residuals,residuals)

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
        return np.dot(residuals, residuals)


    @staticmethod
    def addArgs(parser):
        parser.add_argument("--sklearn", action="store_true",
            help="use sklearn linear regression instead of scipy lstsq")
        ## scipy fit options
        parser.add_argument("--max-iter", type=int, default=100,
            help="max number of iterations to use in lsqr")
        parser.add_argument("--atol", type=float, default=1e-4,
            help="a stopping tolerance")
        parser.add_argument("--btol", type=float, default=1e-8,
            help="b stopping tolerance")