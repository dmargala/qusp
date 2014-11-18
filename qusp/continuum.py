"""
Provides classes that represent quasar continuum objects.
"""

import abc
 
import scipy.interpolate
import numpy as np

import qusp

class Continuum(object):
    """
    Abstract base class for quasar continuum objects.
    """
    __metaclass__  = abc.ABCMeta
 
    @abc.abstractmethod
    def get_continuum(self, target, combined):
        """
        Returns a SpectralFluxDensity object that represent's the specified target's unabsorbed continuum.

        Args:
            target (`qusp.target.Target`): the target
            combined (`qusp.spectrum.BOSSSpectrum`): the target's combined spectrum
        """

class LinearFitContinuum(Continuum):
    """
    Abstract base class for quasar continuum objects.
    """
    def __init__(self, specfits):
        import h5py
        # read data sets from specified specfits file
        self.specfits = h5py.File(specfits)
        self.targets = self.specfits['targets'].value
        self.redshifts = self.specfits['redshifts'].value
        self.amp = self.specfits['amplitude'].value
        self.nu = self.specfits['nu'].value   
        self.rest_wave_centers = self.specfits['restWaveCenters'].value
        self.obs_wave_centers = self.specfits['obsWaveCenters'].value
        self.continuum = self.specfits['continuum'].value
        self.transmission = self.specfits['transmission'].value 
        self.tiltwave = self.specfits['nu'].attrs['tiltwave']
        # create interpolated transmission and continuum functions
        self.trans_interp = scipy.interpolate.UnivariateSpline(self.obs_wave_centers, self.transmission, s=0)
        self.cont_interp = scipy.interpolate.UnivariateSpline(self.rest_wave_centers, self.continuum, s=0)

    def get_continuum(self, target, combined):
        """
        Returns a SpectralFluxDensity object that represent's the specified target's unabsorbed continuum.

        Args:
            target (`qusp.target.Target`): the target
            combined (`qusp.spectrum.BOSSSpectrum`): the target's combined spectrum

        Raises:
            ValueError: if target is not found in fit results.
        """
        # make sure we the request target exists
        if not target['target'] in self.targets:
            raise ValueError('Target not found in specified continuum results.')
        target_index = np.argmax(target['target'] == self.targets)
        assert target['z'] == self.redshifts[target_index]

        # save target's amplitude and spectral tilt
        target['nu'] = self.nu[target_index]
        target['amp'] = self.amp[target_index]

        # build the observed continuum from fit results
        redshifted_waves = self.obs_wave_centers/(1+target['z'])
        rest_continuum = target['amp']*(redshifted_waves/self.tiltwave)**target['nu']*self.cont_interp(redshifted_waves)
        obs_continuum = rest_continuum/(1+target['z'])*self.trans_interp(self.obs_wave_centers)

        # return SpectralFluxDensity representation of the observed continuum
        return qusp.SpectralFluxDensity(self.obs_wave_centers, obs_continuum)

class MeanFluxContinuum(Continuum):
    """
    A simple continuum estimate calculated using the mean flux for a quasar.

    Args:
        wave_min (float): Optional rest frame wavelength for lower bound of mean
            flux calculation.
        wave_max (float): Optional rest frame wavelength for upper bound of mean
            flux calculation.
    """
    def __init__(self, wave_min=None, wave_max=None):
        try:
            self.wave_min = qusp.wavelength.Wavelength(wave_min)
        except TypeError:
            self.wave_min = wave_min
        try: 
            self.wave_max = qusp.wavelength.Wavelength(wave_max)
        except TypeError:
            self.wave_max = wave_max

    def get_continuum(self, target, combined):
        """
        Returns a SpectralFluxDensity object that represent's the specified target's unabsorbed continuum.

        Args:
            target (`qusp.target.Target`): the target
            combined (`qusp.spectrum.BOSSSpectrum`): the target's combined spectrum

        Raises:
            ValueError: if mean flux <= 0
        """
        # determine wavelength range
        if self.wave_min is None:
            wave_min = combined.wavelength[0]
        else:
            wave_min = self.wave_min.observed(target['z'])
        if self.wave_max is None:
            wave_max = combined.wavelength[-1]
        else:
            wave_max = self.wave_max.observed(target['z'])
        # calculate mean flux in wavelength range
        mean_flux = combined.mean_flux(wave_min, wave_max)
        if mean_flux <= 0:
            raise ValueError('mean_flux <= 0')
        continuum = mean_flux*np.ones_like(combined.wavelength)
        return qusp.SpectralFluxDensity(combined.wavelength, continuum)

