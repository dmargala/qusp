"""
Provides support for modeling a universal quasar continuum.

Test laptop dev using::

    time ./examples/fitspec.py --boss-root ~/data/boss -i test.txt -o fitspec-test -n 100 --verbose --sklearn --unweighted --z-col 3 --sn-col 5 --save-model --random
    ./examples/plotfitspec.py --boss-root ~/data/boss -i fitspec-test.hdf5 -o output/fitspec-test --force-y --save-model --examples 0 1 2 3 4

Test on darkmatter using::

    time ./examples/fitspec.py --boss-root /data/boss -i sn-sorted.txt -o fitspec-test -n 1000 --verbose --sklearn --unweighted --z-col 3 --sn-col 5 --save-model --absscale 1 --random --tiltweight 1 --restnormweight 1 --obsnormweight 1e-1
    ./examples/plotfitspec.py --boss-root /data/boss -i fitspec-test.hdf5 -o output/fitspec-test --force-y --save-model --examples 150 300 450 600 750

--------------

Profile on darkmatter::

    python -m cProfile -o profile-test.out ./examples/fitspec.py --boss-root /data/boss -i sn-sorted.txt -o specfits-test --verbose --sklearn --unweighted --z-col 3 --sn-col 5 --fix-norm --fix-tilt --obsnormweight 1e-2 --restnormweight 1 -n 10000

List top function calls by time::

    import pstats
    p = pstats.Stats('profile-test.out')
    p.sort_stats('time').print_stats(10)

Generate call tree::

    gprof2dot -f pstats profile-test.out | dot -Tpng -o profile-test.png

--------------

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
    def __init__(self, transmission_min, transmission_max, continuum_min, continuum_max, continuum_nparams, tiltwave,
                 absorption_min, absorption_max, absorption_modelexp, absorption_scale, continuum=None, verbose=False):
        """
        Initializes a linearized quasar continuum model using the specified
        parameter limits and values.

        Args:
            transmission_min (float): minimum observed frame wavelength bin center of
                transmission model.
            transmission_max (float): maximum observed frame wavelength bin center of
                transmission model.
            continuum_min (float): minimum rest frame wavelength bin center of
                continuum model.
            continuum_max (float): maximum rest frame wavelength bin center of
                continuum model.
            continuum_nparams (int): number of rest frame bins of continuum model.
            tiltwave (float): pivot wavelength of rest frame spectral tilt.
            absorption_min (float): minimum rest frame wavelength bin center of
                absorption model.
            absorption_max (float): maximum rest frame wavelength bin center of
                absorption model.
            absorption_modelexp (float): exponent of (1+z) factor of absorption model.
            absorption_scale (float): internal scaling of absorption model coefficients.
            verbose (bool, optional): whether or not to print verbose output.
        """
        self.verbose = verbose
        self.model_nparams = 0  # the number of model params (excluding per target params)
        # initialize transmission model params
        assert transmission_max > transmission_min, ('transmission_max must be greater than transmission_min')
        self.transmission_wave_min = transmission_min
        self.transmission_wave_max = transmission_max
        transmission_fid_wave = qusp.wavelength.get_fiducial_wavelength(np.arange(4800))
        self.transmission_wave_min_index = np.argmax(transmission_fid_wave > transmission_min)
        self.transmission_wave_max_index = np.argmax(transmission_fid_wave > transmission_max)+1
        self.transmission_wave_centers = qusp.wavelength.get_fiducial_wavelength(
            np.arange(self.transmission_wave_min_index, self.transmission_wave_max_index))
        self.transmission_nparams = len(self.transmission_wave_centers)
        if verbose:
            print ('Observed frame bin centers span [%.2f:%.2f] with %d bins.' %
                (self.transmission_wave_centers[0], self.transmission_wave_centers[-1], self.transmission_nparams))
        self.model_nparams += self.transmission_nparams
        # initialize continuum model params
        assert (continuum_nparams > 0) or continuum is not None, ('must specify number of continuum params or provide continuum')
        self.continuum = continuum
        if continuum:
            self.continuum_wave_centers = continuum.wavelength
            self.continuum_wave_delta = continuum.wavelength[1]-continuum.wavelength[0]
            self.continuum_wave_min = continuum.wavelength[0]-0.5*self.continuum_wave_delta
            self.continuum_wave_max = continuum.wavelength[-1]+0.5*self.continuum_wave_delta
            self.continuum_nparams = 0
        else:
            assert continuum_max > continuum_min, ('continuum_min must be greater than continuum_max')
            self.continuum_wave_min = continuum_min
            self.continuum_wave_max = continuum_max
            self.continuum_nparams = continuum_nparams
            self.continuum_wave_delta = float(continuum_max-continuum_min)/continuum_nparams
            self.continuum_wave_centers = 0.5*self.continuum_wave_delta + np.linspace(
                continuum_min, continuum_max, continuum_nparams, endpoint=False)

        if verbose:
            print ('Continuum model wavelength bin centers span [%.2f:%.2f] with %d free params.' %
                (self.continuum_wave_centers[0], self.continuum_wave_centers[-1], self.continuum_nparams))

        self.model_nparams += self.continuum_nparams

        # initialize absorption model params
        self.absorption_modelexp = absorption_modelexp
        self.absorption_wave_min = max(absorption_min, self.continuum_wave_min)
        self.absorption_wave_max = min(absorption_max, self.continuum_wave_max)
        self.absorption_wave_min_index = np.argmax(
            self.continuum_wave_centers >= self.absorption_wave_min)
        self.absorption_wave_max_index = np.argmax(
            self.continuum_wave_centers > self.absorption_wave_max)
        self.absorption_wave_centers = self.continuum_wave_centers[self.absorption_wave_min_index:self.absorption_wave_max_index]
        self.absorption_nparams = len(self.absorption_wave_centers)
        self.absorption_scale = absorption_scale
        if verbose:
            if self.absorption_nparams > 0:
                print ('Absorption bin centers span [%.2f:%.2f] with %d bins.' %
                    (self.absorption_wave_centers[0], self.absorption_wave_centers[-1], self.absorption_nparams))
            else:
                print 'No absorption params'
        self.model_nparams += self.absorption_nparams
        # spectral tilt pivot wavelength
        self.tiltwave = tiltwave
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
        transmission_fiducial_indices = fiducial_pixel_offset+np.arange(len(wave))
        transmission_fid_wave = qusp.wavelength.get_fiducial_wavelength(
            transmission_fiducial_indices)
        # map pixels to rest frame wavelength grid
        continuum_wave = transmission_fid_wave/(1+redshift)
        continuum_indices = np.floor(
            (continuum_wave - self.continuum_wave_min)/self.continuum_wave_delta).astype(int)
        # trim ranges to valid data
        valid_pixels = np.all((
            transmission_fiducial_indices >= self.transmission_wave_min_index,
            transmission_fiducial_indices < self.transmission_wave_max_index,
            continuum_indices < len(self.continuum_wave_centers),
            continuum_indices >= 0,
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
        transmission_indices = transmission_fiducial_indices[valid_pixels]-self.transmission_wave_min_index
        assert np.amax(transmission_indices) < self.transmission_nparams, (
            'Invalid obsmodel index value')
        transmission_block = scipy.sparse.coo_matrix(
            (np.ones(npixels), (np.arange(npixels), transmission_indices)),
            shape=(npixels, self.transmission_nparams))
        obs_blocks.append(transmission_block)
        # build continuum model param block
        continuum_indices = continuum_indices[valid_pixels]
        assert np.amax(continuum_indices) < len(self.continuum_wave_centers), (
            'Invalid rest model index value')
        if self.continuum:
            try:
                yvalues -= np.log(self.continuum(self.continuum_wave_centers[continuum_indices]))
            except ValueError, e:
                print continuum_wave[valid_pixels]
                print self.continuum_wave_centers[continuum_indices]
                raise e
        else:
            continuum_block = scipy.sparse.coo_matrix(
                (np.ones(npixels), (np.arange(npixels), continuum_indices)),
                shape=(npixels, self.continuum_nparams))
            obs_blocks.append(continuum_block)
        # build absorption model param block
        absorption_wave_min_index = np.argmax(continuum_indices == self.absorption_wave_min_index)
        absorption_wave_max_index = np.argmax(continuum_indices == self.absorption_wave_max_index)
        # check if any of the this observation has pixels in the relevant
        # absorption range
        if absorption_wave_max_index > absorption_wave_min_index:
            absorption_rows = np.arange(npixels)[absorption_wave_min_index:absorption_wave_max_index]
            absorption_cols = (continuum_indices[absorption_wave_min_index:absorption_wave_max_index] -
                        self.absorption_wave_min_index)
            assert np.amax(absorption_cols) < self.absorption_nparams, (
                'Invalid abs param index value')
            absorption_coefficients = -np.ones(len(absorption_cols))*(
                np.power(1+redshift, self.absorption_modelexp)/self.absorption_scale)
            absorption_block = scipy.sparse.coo_matrix(
                (absorption_coefficients, (absorption_rows, absorption_cols)),
                shape=(npixels, self.absorption_nparams))
            obs_blocks.append(absorption_block)
        elif self.absorption_nparams > 0:
            absorption_block = scipy.sparse.coo_matrix(
                (npixels, self.absorption_nparams))
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
            np.log(self.continuum_wave_centers[continuum_indices]/self.tiltwave))
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

    def add_transmission_constraint(self, yvalue, wavemin, wavemax, weight):
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
        waves = self.transmission_wave_centers
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

    def add_continuum_constraint(self, yvalue, wavemin, wavemax, weight):
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
        if self.continuum:
            return
        # rest model params are after obs model params
        offset = self.transmission_nparams
        # find range of continuum model params in constraint window
        waves = self.continuum_wave_centers
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
        offset = self.transmission_nparams + self.continuum_nparams + self.absorption_nparams + self.amplitude.count(None)
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
        amplitude_nparams = self.amplitude.count(None)
        amplitude_counter = 0
        tilt_nparams = self.tilt.count(None)
        tilt_counter = 0
        for obs_index in range(self.ntargets):
            npixels = self.obs_blocks[obs_index][0].shape[0]
            # add norm block
            # if there are no free amplitude parameters then we wont need any blocks
            if amplitude_nparams > 0:
                amp = self.amplitude[obs_index]
                if amp is None:
                    amp_indices = amplitude_counter*np.ones(npixels)
                    amplitude_counter += 1
                    amplitude_block = scipy.sparse.coo_matrix((self.norm_coefficients[obs_index],
                        (np.arange(npixels), amp_indices)), shape=(npixels, amplitude_nparams))
                else:
                    amplitude_block = scipy.sparse.coo_matrix((npixels, amplitude_nparams))
                self.obs_blocks[obs_index].append(amplitude_block)
            # add tilt block
            if tilt_nparams > 0:
                tilt = self.tilt[obs_index]
                if tilt is None:
                    tilt_indices = tilt_counter*np.ones(npixels)
                    tilt_counter += 1
                    tilt_block = scipy.sparse.coo_matrix((self.tilt_coefficients[obs_index],
                        (np.arange(npixels), tilt_indices)), shape=(npixels, tilt_nparams))
                else:
                    tilt_block = scipy.sparse.coo_matrix((npixels, tilt_nparams))
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

        #print self.model[:self.model_npixels].sum(axis=0)[0,4437:4437+1000]

        # concatenate y values
        yvalues = np.concatenate(self.modelyvalues)
        self.modely = yvalues

        if self.verbose:
            print 'Number of transmission model params: %d' % self.transmission_nparams
            print 'Number of continuum model params: %d' % self.continuum_nparams
            print 'Number of absorption model params: %d' % self.absorption_nparams
            print 'Number of targets: %d' % self.ntargets
            print 'Number of amplitude params: %d' % amplitude_nparams
            print 'Number of tilt params: %d' % tilt_nparams
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
        results['transmission'] = np.exp(soln[offset:offset+self.transmission_nparams])
        offset += self.transmission_nparams
        # continuum: transform logC -> C
        if self.continuum:
            results['continuum'] = self.continuum.values
        else:
            results['continuum'] = np.exp(soln[offset:offset+self.continuum_nparams])
        offset += self.continuum_nparams
        # absorption
        results['absorption'] = soln[offset:offset+self.absorption_nparams]*(
            self.absorption_scale)
        offset += self.absorption_nparams
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

    def save(self, filename, soln, args, save_model=True, save_chisq=True):
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
            save_chisq (bool, optional): whether or not to save per 
                observation chisq values. Defaults to True.

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
        outfile.create_dataset('obsWaveCenters', data=self.transmission_wave_centers)
        outfile.create_dataset('restWaveCenters', data=self.continuum_wave_centers)
        # save transmission model params and relevant info
        transmission = outfile.create_dataset(
            'transmission', data=results['transmission'])
        transmission.attrs['normmin'] = args.transmission_normmin
        transmission.attrs['normmax'] = args.transmission_normmax
        transmission.attrs['normweight'] = args.transmission_normweight
        # save continuum model params and relevant info
        continuum = outfile.create_dataset(
            'continuum', data=results['continuum'])
        continuum.attrs['normmin'] = args.continuum_normmin
        continuum.attrs['normmax'] = args.continuum_normmax
        continuum.attrs['normweight'] = args.continuum_normweight
        # save absorption model params and relevant info
        absorption = outfile.create_dataset(
            'absorption', data=results['absorption'])
        absorption.attrs['minRestIndex'] = self.absorption_wave_min_index
        absorption.attrs['maxRestIndex'] = self.absorption_wave_max_index
        absorption.attrs['absmodelexp'] = self.absorption_modelexp
        # save amplitude params and relevant info
        outfile.create_dataset('amplitude', data=results['amplitude'])
        # save spectral tilt params and relevant info
        tilt = outfile.create_dataset('nu', data=results['nu'])
        tilt.attrs['tiltwave'] = args.tiltwave
        tilt.attrs['tiltweight'] = args.tiltweight
        # save per-obs chisqs
        if save_chisq:
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
        parser.add_argument("--transmission-min", type=float, default=3600,
            help="transmission model wavelength minimum")
        parser.add_argument("--transmission-max", type=float, default=10000,
            help="transmission model wavelength maximum")
        parser.add_argument("--transmission-normmin", type=float, default=3600,
            help="obsframe wavelength to normalize at")
        parser.add_argument("--transmission-normmax", type=float, default=10000,
            help="obsframe window size +/- on each side of obsnorm wavelength")
        parser.add_argument("--transmission-normweight", type=float, default=1,
            help="norm constraint weight")
        # continuum model wavelength grid options
        parser.add_argument("--continuum-min", type=float, default=900,
            help="rest wavelength minimum")
        parser.add_argument("--continuum-max", type=float, default=2900,
            help="rest wavelength maximum")
        parser.add_argument("--continuum-nparams", type=int, default=1000,
            help="number of restframe bins")
        parser.add_argument("--continuum-normmin", type=float, default=1275,
            help="restframe window normalization minimum")
        parser.add_argument("--continuum-normmax", type=float, default=1285,
            help="restframe window normalization maximum")
        parser.add_argument("--continuum-normweight", type=float, default=1,
            help="norm constraint weight")
        # absorption model parameter options
        parser.add_argument("--absorption-min", type=float, default=900,
            help="absorption min wavelength (rest frame)")
        parser.add_argument("--absorption-max", type=float, default=1400,
            help="absoprtion max wavelength (rest frame)")
        parser.add_argument("--absorption-modelexp", type=float, default=3.92,
            help="absorption model (1+z) factor exponent")
        parser.add_argument("--absorption-scale", type=float, default=1,
            help="scale absorption params in fit")
        # spectral tilt parameter options
        parser.add_argument("--tiltwave", type=float, default=1280,
            help="spectral tilt pivot wavelength")
        parser.add_argument("--tiltweight", type=float, default=1e-2,
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
