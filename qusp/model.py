"""
Provides support for modeling a universal quasar continuum.

Test laptop dev using::

    time ./examples/fitspec.py --boss-root ~/data/boss -i test.txt -o fitspec-test -n 100 --verbose --sklearn --unweighted --z-col 3 --sn-col 5 --save-model --absscale 1 --random
    ./examples/plotfitspec.py --boss-root ~/data/boss -i fitspec-test.hdf5 -o output/fitspec-test --force-y --save-model --examples 0 1 2 3 4

Test on darkmatter using::

    time ./examples/fitspec.py --boss-root /data/boss -i sn-sorted.txt -o fitspec-test -n 1000 --verbose --sklearn --unweighted --z-col 3 --sn-col 5 --save-model --absscale 1 --random --tiltweight 1 --restnormweight 1 --obsnormweight 1e-1
    ./examples/plotfitspec.py --boss-root /data/boss -i fitspec-test.hdf5 -o output/fitspec-test --force-y --save-model --examples 150 300 450 600 750

"""

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

        Args:
            obsmin (float): minimum observed frame wavelength bin center of
                transmission model.
            obsmax (float): maximum observed frame wavelength bin center of
                transmission model.
            restmin (float): minimum rest frame wavelength bin center of
                continuum model.
            restmax (float): maximum rest frame wavelength bin center of
                continuum model.
            nrestbins (int): number of rest frame bins of continuum model.
            tiltwave (float): pivot wavelength of rest frame spectral tilt.
            absmin (float): minimum rest frame wavelength bin center of
                absorption model.
            absmax (float): maximum rest frame wavelength bin center of
                absorption model.
            absmodelexp (float): exponent of (1+z) factor of absorption model.
            absscale (float): internal scaling of absorption model coefficients.
            verbose (bool, optional): whether or not to print verbose output.
        """
        self.verbose = verbose
        # initialize transmission model params
        assert obsmax > obsmin, ('obsmax must be greater than obsmin')
        self.obs_wave_min = obsmin
        self.obs_wave_max = obsmax
        obs_fid_wave = qusp.wavelength.get_fiducial_wavelength(np.arange(4800))
        self.obs_wave_min_index = np.argmax(obs_fid_wave > obsmin)
        self.obs_wave_max_index = np.argmax(obs_fid_wave > obsmax)+1
        self.obs_wave_centers = qusp.wavelength.get_fiducial_wavelength(
            np.arange(self.obs_wave_min_index, self.obs_wave_max_index))
        self.obs_nparams = len(self.obs_wave_centers)
        if verbose:
            print ('Observed frame bin centers span [%.2f:%.2f] with %d bins.' %
                   (self.obs_wave_centers[0], self.obs_wave_centers[-1],
                    self.obs_nparams))
        # initialize continuum model params
        assert restmax > restmin, ('restmin must be greater than restmax')
        self.rest_wave_min = restmin
        self.rest_wave_max = restmax
        self.rest_nparams = nrestbins
        self.rest_wave_delta = float(restmax-restmin)/nrestbins
        self.rest_wave_centers = 0.5*self.rest_wave_delta + np.linspace(
            restmin, restmax, nrestbins, endpoint=False)
        if verbose:
            print ('Rest frame bin centers span [%.2f:%.2f] with %d bins.' %
                   (self.rest_wave_centers[0], self.rest_wave_centers[-1],
                    self.rest_nparams))
        # initialize absorption model params
        self.absmodelexp = absmodelexp
        self.abs_wave_min = max(absmin, restmin)
        self.abs_wave_max = min(absmax, restmax)
        self.abs_wave_min_index = np.argmax(
            self.rest_wave_centers >= self.abs_wave_min)
        self.abs_wave_max_index = np.argmax(
            self.rest_wave_centers > self.abs_wave_max)
        self.abs_wave_centers = self.rest_wave_centers[self.abs_wave_min_index:self.abs_wave_max_index]
        self.abs_nparams = len(self.abs_wave_centers)
        self.abs_scale = absscale
        if verbose:
            if self.abs_nparams > 0:
                print ('Absorption bin centers span [%.2f:%.2f] with %d bins.' %
                       (self.abs_wave_centers[0], self.abs_wave_centers[-1],
                        self.abs_nparams))
            else:
                print 'No absorption params'
        # spectral tilt pivot wavelength
        self.tiltwave = tiltwave
        # the number of model params (excluding per target params)
        self.model_nparams = (self.obs_nparams + self.rest_nparams +
                              self.abs_nparams)
        if verbose:
            print 'Fit model initialized with %d model params.\n' % (
                self.model_nparams)
        # sparse matrix entry holders
        self.modelyvalues = list()
        self.obs_blocks = list()
        self.constraint_blocks = list()
        self.amplitude = list()
        self.norm_coefficients = list()
        self.tilt = list()
        self.tilt_coefficients = list()
        # store finalized model pieces
        self.model = None
        self.modely = None
        # model stats
        self.model_npixels = 0
        self.ntargets = 0
        self.model_nconstraints = 0

    def add_observation(self, target, flux, wave, ivar, unweighted=True):
        """
        Adds an observation to be fit. Returns the number of pixels added.
        The provided target argument must have an attribute named 'z' with
        the target's redshift.

        Note:
            Weighted fit not yet implemented.

        Args:
            target (:class:`qusp.target.Target`): a Target object
            flux (numpy.array): flux array
            wave (numpy.array): wavelength array
            ivar (numpy.array): ivar array
            unweighted (bool, optional): ignore pixel variances.
                Defaults to True.

        Returns:
            npixels (int): number of pixels added to model from this observation

        """
        assert unweighted, ('Weighted fit not implemented yet...')
        try:
            redshift = target['z']
        except KeyError:
            print 'Target does not have z attribute.'
            raise
        # this spectrum's co-add wavelength axis pixel offset
        fiducial_pixel_offset = qusp.wavelength.get_fiducial_pixel_index_offset(
            np.log10(wave[0]))
        # map pixels to observed frame wavelength grid
        obs_fiducial_indices = fiducial_pixel_offset+np.arange(len(wave))
        obs_fid_wave = qusp.wavelength.get_fiducial_wavelength(
            obs_fiducial_indices)
        # map pixels to rest frame wavelength grid
        rest_wave = obs_fid_wave/(1+redshift)
        rest_indices = np.floor(
            (rest_wave - self.rest_wave_min)/self.rest_wave_delta).astype(int)
        # trim ranges to valid data
        valid_pixels = np.all((
            obs_fiducial_indices >= self.obs_wave_min_index,
            obs_fiducial_indices < self.obs_wave_max_index,
            rest_indices < self.rest_nparams,
            rest_indices >= 0,
            flux > 0, ivar > 0), axis=0)
        npixels = np.sum(valid_pixels)
        # skip target if no valid pixels
        if npixels <= 0:
            if self.verbose:
                print 'No good pixels for target %s (z=%.2f)' % (
                    target['target'], redshift)
            return 0
        # initialize y values to logf + log(1+redshift)
        yvalues = np.log(flux[valid_pixels]) + np.log(1+redshift)
        ##### TODO: compute weights
        #if unweighted:
        #     weights = np.ones(npixels)
        # else:
        #     weights = ivar[valid_pixels]
        # sqrtw = np.sqrt(weights)
        # yvalues = sqrtw*yvalues
        ###################
        obs_blocks = []
        # build transmission model param block
        obs_indices = obs_fiducial_indices[valid_pixels]-self.obs_wave_min_index
        assert np.amax(obs_indices) < self.obs_nparams, (
            'Invalid obsmodel index value')
        transmission_block = scipy.sparse.coo_matrix(
            (np.ones(npixels), (np.arange(npixels), obs_indices)),
            shape=(npixels, self.obs_nparams))
        obs_blocks.append(transmission_block)
        # build continuum model param block
        rest_indices = rest_indices[valid_pixels]
        assert np.amax(rest_indices) < self.rest_nparams, (
            'Invalid rest model index value')
        continuum_block = scipy.sparse.coo_matrix(
            (np.ones(npixels), (np.arange(npixels), rest_indices)),
            shape=(npixels, self.rest_nparams))
        obs_blocks.append(continuum_block)
        # build absorption model param block
        abs_wave_min_index = np.argmax(rest_indices == self.abs_wave_min_index)
        abs_wave_max_index = np.argmax(rest_indices == self.abs_wave_max_index)
        # check if any of the this observation has pixels in the relevant
        # absorption range
        if abs_wave_max_index > abs_wave_min_index:
            abs_rows = np.arange(npixels)[abs_wave_min_index:abs_wave_max_index]
            abs_cols = (rest_indices[abs_wave_min_index:abs_wave_max_index] -
                        self.abs_wave_min_index)
            assert np.amax(abs_cols) < self.abs_nparams, (
                'Invalid abs param index value')
            abs_coefficients = -np.ones(len(abs_cols))*(
                np.power(1+redshift, self.absmodelexp)/self.abs_scale)
            absorption_block = scipy.sparse.coo_matrix(
                (abs_coefficients, (abs_rows, abs_cols)),
                shape=(npixels, self.abs_nparams))
            obs_blocks.append(absorption_block)
        elif self.abs_nparams > 0:
            absorption_block = scipy.sparse.coo_matrix(
                (npixels, self.abs_nparams))
            obs_blocks.append(absorption_block)
        else:
            # no absorption parameters, do nothing
            pass
        # process amplitude params
        try:
            amp = target['amp']
            yvalues -= np.log(amp)
            self.norm_coefficients.append(None)
        except KeyError:
            amp = None
            self.norm_coefficients.append(np.ones(npixels))
            self.model_nparams += 1
        self.amplitude.append(amp)
        # process spectral tilt params
        tilt_coefficients = (
            np.log(self.rest_wave_centers[rest_indices]/self.tiltwave))
        try:
            tilt = target['nu']
            yvalues -= tilt*tilt_coefficients
            self.tilt_coefficients.append(None)
        except KeyError:
            tilt = None
            self.tilt_coefficients.append(tilt_coefficients)
            self.model_nparams += 1
        self.tilt.append(tilt)
        # append this observation's blocks to the model
        self.obs_blocks.append(obs_blocks)
        # append this observation's y values to the model
        self.modelyvalues.append(yvalues)
        # update model stats
        self.model_npixels += npixels
        self.ntargets += 1
        # return number of pixels added from this observation
        return npixels

    def add_obs_constraint(self, yvalue, wavemin, wavemax, weight):
        """
        Adds a constraint equation for each of the transmission model params
        between the specified observed frame wavelengths.

        Args:
            yvalue (float): y value of constraint equation.
            wavemin (float): minimum observed frame wavelength bin center
                to constrain.
            wavemax (float): maxmimum observed frame wavelength bin center
                to constrain.
            weight (float): weight to apply to constraint equation.
        """
        # transmission model params are first
        offset = 0
        # find range of transmission model params in constraint window
        waves = self.obs_wave_centers
        wave_index_range = np.arange(np.argmax(waves > wavemin), np.argmax(waves > wavemax)+1)
        # number of transmission parameters in constraint window
        transmission_nparams = len(wave_index_range)
        # scale weight
        weight *= self.model_npixels/transmission_nparams
        # build constraint block
        coefs = weight*np.ones(transmission_nparams)
        rows = np.arange(transmission_nparams)
        cols = offset+wave_index_range
        block = scipy.sparse.coo_matrix(
            (coefs, (rows, cols)),
            shape=(transmission_nparams, self.model_nparams))
        if self.verbose:
            print 'Adding constraint: %.2g*logT([%.2f:%.2f]) = %.1f (%d logT params [%d:%d])' % (
                weight, waves[wave_index_range[0]], waves[wave_index_range[-1]], yvalue,
                transmission_nparams, wave_index_range[0], wave_index_range[-1])
        # append constraint block and update number of constraint equations
        self.constraint_blocks.append(block)
        self.model_nconstraints += transmission_nparams
        yvalues = yvalue*np.ones(transmission_nparams)
        self.modelyvalues.append(yvalues)

    def add_rest_constraint(self, yvalue, wavemin, wavemax, weight):
        """
        Adds a constraint equation on the geometric mean of the continuum model
        in between the specified rest frame wavelengths.

        Args:
            yvalue (float): y value of constraint equation.
            wavemin (float): minimum rest frame wavelength bin center
                to constrain.
            wavemax (float): maxmimum rest frame wavelength bin center
                to constrain.
            weight (float): weight to apply to constraint equation.
        """
        # rest model params are after obs model params
        offset = self.obs_nparams
        # find range of continuum model params in constraint window
        waves = self.rest_wave_centers
        wave_index_range = np.arange(np.argmax(waves > wavemin), np.argmax(waves > wavemax))
        # number of continuum parameters in constraint window
        continuum_nparams = len(wave_index_range)
        # scale weight
        weight *= self.model_npixels/continuum_nparams
        # build constraint block
        coefs = weight*np.ones(continuum_nparams)
        rows = np.zeros(continuum_nparams)
        cols = offset+wave_index_range
        block = scipy.sparse.coo_matrix(
            (coefs, (rows, cols)), shape=(1, self.model_nparams))
        if self.verbose:
            print 'Adding constraint: sum(%.2g*logC([%.2f:%.2f])) = %.1f (%d logC params [%d:%d])' % (
                weight, waves[wave_index_range[0]], waves[wave_index_range[-1]], yvalue,
                continuum_nparams, wave_index_range[0], wave_index_range[-1])
        # append constraint block and update number of constraint equations
        self.constraint_blocks.append(block)
        self.model_nconstraints += 1
        self.modelyvalues.append([yvalue])

    def add_tilt_constraint(self, weight):
        """
        Adds a constraint equation on the mean of the non-fixed spectral tilt
        params.

        Args:
            weight (float): weight to apply to constraint equation.
        """
        # calculate tilt block column offset, tilt params are after
        # normalization params
        offset = self.obs_nparams + self.rest_nparams + self.abs_nparams + self.amplitude.count(None)
        # count number of free tilt params
        tilt_nparams = self.tilt.count(None)
        # scale weight by the total number of pixels and the fraction of targets with free tilt params
        weight *= self.model_npixels*tilt_nparams/self.ntargets
        # build constraint block
        coefs = weight*np.ones(tilt_nparams)/tilt_nparams
        rows = np.zeros(tilt_nparams)
        cols = offset+np.arange(tilt_nparams)
        block = scipy.sparse.coo_matrix(
            (coefs, (rows, cols)), shape=(1, self.model_nparams))
        if self.verbose:
            print 'Adding constraint: %.2g*sum(nu) = 0 (%d tilt params)' % (
                weight, tilt_nparams)
        # append constraint block and update number of constraint equations
        self.constraint_blocks.append(block)
        self.model_nconstraints += 1
        self.modelyvalues.append([0])

    def finalize(self):
        """
        Does final assembly of the sparse matrix representing the model.
        """
        # pass through each observation and do final assembly of
        # target param blocks
        for obs_index in range(self.ntargets):
            npixels = self.obs_blocks[obs_index][0].shape[0]
            # add norm block
            if self.amplitude.count(None) > 0:
                amp = self.amplitude[obs_index]
                if amp is None:
                    amp_indices = self.amplitude[:obs_index].count(None)*(
                        np.ones(npixels))
                    amplitude_block = scipy.sparse.coo_matrix(
                        (self.norm_coefficients[obs_index],
                        (np.arange(npixels), amp_indices)),
                        shape=(npixels, self.amplitude.count(None)))
                else:
                    amplitude_block = scipy.sparse.coo_matrix(
                        (npixels, self.amplitude.count(None)))
                self.obs_blocks[obs_index].append(amplitude_block)
            # add tilt block
            if self.tilt.count(None) > 0:
                tilt = self.tilt[obs_index]
                if tilt is None:
                    tilt_indices = self.tilt[:obs_index].count(None)*(
                        np.ones(npixels))
                    tilt_block = scipy.sparse.coo_matrix(
                        (self.tilt_coefficients[obs_index],
                        (np.arange(npixels), tilt_indices)),
                        shape=(npixels, self.tilt.count(None)))
                else:
                    tilt_block = scipy.sparse.coo_matrix(
                        (npixels, self.tilt.count(None)))
                self.obs_blocks[obs_index].append(tilt_block)
        # combine blocks from all observations
        obs_block = scipy.sparse.bmat(self.obs_blocks)
        # combine blocks from all constraints
        constraint_block = scipy.sparse.vstack(self.constraint_blocks)
        # comebine observations and constraints
        final_model = scipy.sparse.vstack([obs_block, constraint_block])

        assert final_model.shape[0] == self.model_npixels + self.model_nconstraints
        assert final_model.shape[1] == self.model_nparams

        self.model = final_model.tocsc()

        # concatenate y values
        yvalues = np.concatenate(self.modelyvalues)
        self.modely = yvalues

        if self.verbose:
            print 'Number of transmission model params: %d' % self.obs_nparams
            print 'Number of continuum model params: %d' % self.rest_nparams
            print 'Number of absorption model params: %d' % self.abs_nparams
            print 'Number of targets: %d' % self.ntargets
            print 'Number of amplitude params: %d' % self.amplitude.count(None)
            print 'Number of tilt params: %d' % self.tilt.count(None)
            print ''
            print 'Total number of model params: %d' % self.model_nparams
            print 'Total number of flux measurements: %d (%d constraints)' % (
                self.model_npixels, self.model_nconstraints)
            print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)\n' % (
                self.model.data.nbytes, self.model.indptr.nbytes, self.model.indices.nbytes)

    def get_model(self):
        """
        Returns the assembled model matrix and corresponding y values

        Returns:
            A tuple of ``(model, yvalues)``, where *model* is a
                :class:`scipy.sparse.csc_matrix` and *yvalues* is a
                :class:`numpy.array`.
        """
        if self.model is None:
            self.finalize()
        return self.model, self.modely

    def get_results(self, soln):
        """
        Converts the provided solution to model params. Transforms log params
        to linear and inserts fixed model params.

        Returns:
            results (dict): a dictionary of model params
        """
        assert self.model is not None, (
            'Can\'t request results before model assembly.')
        assert len(soln) == self.model_nparams, (
            'Size of soln does not match model.')
        results = dict()
        offset = 0
        # transmission: transform logT -> T
        results['transmission'] = np.exp(soln[offset:offset+self.obs_nparams])
        offset += self.obs_nparams
        # continuum: transform logC -> C
        results['continuum'] = np.exp(soln[offset:offset+self.rest_nparams])
        offset += self.rest_nparams
        # absorption
        results['absorption'] = soln[offset:offset+self.abs_nparams]*(
            self.abs_scale)
        offset += self.abs_nparams
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
        tilt = list()
        tiltindex = 0
        for tiltvalue in self.tilt:
            if tiltvalue is None:
                tiltvalue = soln[offset+tiltindex]
                tiltindex += 1
            tilt.append(tiltvalue)
        results['nu'] = tilt
        offset += self.tilt.count(None)
        # return dictionary of parameters
        return results

    def get_chisq(self, soln):
        """
        Calculates the chi-squared between the specified solution and
        the model y values.

        Args:
            soln (numpy.array): model parameter solution array

        Returns:
            chisq (float): value
        """
        assert self.model is not None, (
            'Can\'t request chisq before model assembly.')
        # calculate residuals
        residuals = self.modely - self.model.dot(soln)
        # return chisq
        return np.dot(residuals, residuals)/len(residuals)

    def get_obs_chisq(self, soln, obs_index):
        """
        Returns the chi-squared value of the specified observation index,
        *obs_index*, using the specified soln.

        Args:
            soln (numpy.array): model parameter solution array
            obs_index (int): observation index

        Returns:
            chisq (float): value
        """
        assert self.model is not None, (
            'Can\'t request chisq before model assembly.')
        # check soln size
        assert len(soln) == self.model_nparams, (
            'Size of soln does not match model.')
        # pick out y value entries for the requested observation
        yvalues = self.modelyvalues[obs_index]
        # pick out model rows for the requested observation
        model = scipy.sparse.hstack(self.obs_blocks[obs_index]).tocsc()
        # calculate residuals
        residuals = yvalues - model.dot(soln)
        # return chisq
        return np.dot(residuals, residuals)/len(residuals)

    def save(self, filename, soln, args, save_model=True):
        """
        Saves soln to the specified filename as an hdf5 file. Parsed results
        and fit meta data are also saved. Use the saveModel arg to indicate
        whether or not to save the raw data of the sparse matrix model.

        Args:
            filename (str): filename of the hdf5 output to create
            soln (numpy.array): model parameter solution array
            args (argparse.Namespace): argparse argument namespace
            save_model (bool, optional): whether or not to save the model
                matrix and y values. Defaults to True.

        Returns:
            outfile (h5py.File): the output hdf5 file created
        """
        results = self.get_results(soln)
        # open hdf5 output file
        outfile = h5py.File(filename, 'w')
        # save soln
        outfile.create_dataset('soln', data=soln)
        if save_model:
            # save model data
            outfile.create_dataset('model_data', data=self.model.data)
            outfile.create_dataset('model_indices', data=self.model.indices)
            outfile.create_dataset('model_indptr', data=self.model.indptr)
            outfile.create_dataset('model_shape', data=self.model.shape)
            outfile.create_dataset('y', data=self.modely)
        # save wavelength grids
        outfile.create_dataset('obsWaveCenters', data=self.obs_wave_centers)
        outfile.create_dataset('restWaveCenters', data=self.rest_wave_centers)
        # save transmission model params and relevant info
        transmission = outfile.create_dataset(
            'transmission', data=results['transmission'])
        transmission.attrs['normmin'] = args.obsnormmin
        transmission.attrs['normmax'] = args.obsnormmax
        transmission.attrs['normweight'] = args.obsnormweight
        # save continuum model params and relevant info
        continuum = outfile.create_dataset(
            'continuum', data=results['continuum'])
        continuum.attrs['normmin'] = args.restnormmin
        continuum.attrs['normmax'] = args.restnormmax
        continuum.attrs['normweight'] = args.restnormweight
        # save absorption model params and relevant info
        absorption = outfile.create_dataset(
            'absorption', data=results['absorption'])
        absorption.attrs['minRestIndex'] = self.abs_wave_min_index
        absorption.attrs['maxRestIndex'] = self.abs_wave_max_index
        absorption.attrs['absmodelexp'] = self.absmodelexp
        # save amplitude params and relevant info
        outfile.create_dataset('amplitude', data=results['amplitude'])
        # save spectral tilt params and relevant info
        tilt = outfile.create_dataset('nu', data=results['nu'])
        tilt.attrs['tiltwave'] = args.tiltwave
        tilt.attrs['tiltweight'] = args.tiltweight
        # save per-obs chisqs
        chisqs = [self.get_obs_chisq(soln, i) for i in range(self.ntargets)]
        outfile.create_dataset('chisq', data=chisqs)
        # return h5py output file
        return outfile

    @staticmethod
    def add_args(parser):
        """
        Add arguments to the provided command-line parser.

        Args:
            parser (argparse.ArgumentParser): an argument parser
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
            help="restframe window normalization minimum")
        parser.add_argument("--restnormmax", type=float, default=1285,
            help="restframe window normalization maximum")
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
        parser.add_argument(
            "--unweighted", action="store_true",
            help="perform unweighted least squares fit")

    @staticmethod
    def from_args(args):
        """
        Returns a dictionary of constructor parameter values based on the
        parsed args provided.

        Args:
            args (argparse.Namespace): argparse argument namespace

        Returns:
            a dictionary of :class:`ContinuumModel` constructor parameter values
        """
        # Look up the named ContinuumModel constructor parameters.
        pnames = (inspect.getargspec(ContinuumModel.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        args_dict = vars(args)
        # Return a dictionary of constructor parameters provided in args.
        return {key:args_dict[key] for key in set(pnames) & set(args_dict)}
