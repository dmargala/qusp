"""
Provides support for working with BOSS spectra.
"""
import numpy as np

import qusp

class Spectrum(object):
    """
    Represents a BOSS co-added spectrum.
    """
    def __init__(self, wavelength, flux, ivar):
        """
        Initializes a spectrum with the provided wavelength, flux, and
        ivar arrays.

        Args:
            wavelength (numpy.array): wavelength pixel centers.
            flux (numpy.array): flux values.
            ivar (numpy.array): flux inverse variance values.
        """
        self.wavelength = wavelength
        self.flux = flux
        self.ivar = ivar
        self.npixels = len(wavelength)
        self.nzero_ivar_pixels = np.sum(ivar == 0)

    def find_pixel(self, wavelength):
        """
        Returns the corresponding pixel index of the specified wavelength.

        Args:
            wavelength (float): value

        Returns:
            pixelIndex (int): pixel index
        """
        if wavelength <= self.wavelength[0]:
            return -1
        if wavelength >= self.wavelength[-1]:
            return self.npixels-1
        # find first pixel with a central wavelength greater than wavelength
        candidate = np.argmax(self.wavelength >= wavelength)
        # compare wavelength with this pixel's lower boundary
        if wavelength > qusp.wavelength.get_fiducial_wavelength(candidate-0.5):
            return candidate
        else:
            return candidate - 1

    def mean_flux(self, min_wavelength, max_wavelength, ivar_weighting=True):
        """
        Returns the mean flux between the specified wavelengths.
        Use ivar_weighting=False option to turn ignore weights.

        Args:
            min_wavelength (float): minimum wavelength for mean flux calculation
                range.
            max_wavelength (float): maximum wavelength for mean flux calculation
                range.
            ivar_weighting (bool, optional): Whether or not to weight
                calculation using inverse variance.

        Returns:
            the mean flux between `min_wavelength` and `max_wavelength`.
        """
        min_pixel = self.find_pixel(min_wavelength)+1
        max_pixel = self.find_pixel(max_wavelength)
        if min_pixel > max_pixel:
            return 0
        pixels = slice(min_pixel, max_pixel+1)
        # only use "good" pixels
        nonzero = np.nonzero(self.ivar[pixels])
        # use weights?
        weights = self.ivar[pixels][nonzero] if ivar_weighting else (
            np.ones(len(nonzero)))
        # calculate mean
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            return 0
        weighted_flux_sum = np.sum(weights*self.flux[pixels][nonzero])
        return weighted_flux_sum/weights_sum

    def median_signal_to_noise(self, min_wavelength, max_wavelength):
        """
        Returns the median signal to noise ratio between the specified
        wavelengths.

        Args:
            min_wavelength (float): minimum wavelength for median flux
                calculation range.
            max_wavelength (float): maximum wavelength for median flux
                calculation range.

        Returns:
            median (float): the median flux between `min_wavelength` and
                `max_wavelength`.
        """
        min_pixel = self.find_pixel(min_wavelength)+1
        max_pixel = self.find_pixel(max_wavelength)
        if min_pixel > max_pixel:
            return 0
        pixels = slice(min_pixel, max_pixel+1)
        # only use "good" pixels
        nonzero = np.nonzero(self.ivar[pixels])
        if len(nonzero) == 0:
            return 0
        # calculate signal to noise
        signal_to_noise = np.fabs(
            self.flux[pixels][nonzero])*np.sqrt(self.ivar[pixels][nonzero])
        # return median
        return np.median(signal_to_noise)

def read_combined_spectrum(spplate, fiber):
    """
    Returns the combined spectrum of the specified fiber from the provided
    spPlate.

    Args:
        spplate (astropy.io.fits.HDUList): spPlate file
        fiber (int,:class:`qusp.target.Target`): boss target's fiberid, or a
            :class:`qusp.target.Target` object.

    Returns:
        spectrum (:class:`Spectrum`): a :class:`Spectrum` object of `fiber`
            of `spplate`.
    """
    # those pesky fiber numbers start at 1 but the fits table is 0-indexed
    if type(fiber) is qusp.target.Target:
        index = fiber['fiber'] - 1
    else:
        index = fiber - 1

    coeff0 = spplate[0].header['COEFF0']
    coeff1 = spplate[0].header['COEFF1']

    flux = spplate[0].data[index]
    ivar = spplate[1].data[index]
    and_mask = spplate[2].data[index]
    or_mask = spplate[3].data[index]
    pixel_dispersion = spplate[4].data[index]
    # Calculate the wavelength sequence to use.
    npixels = len(flux)
    wavelength = np.power(10, coeff0 + coeff1*np.arange(0, npixels))

    # Build the spectrum (no masking yet).
    spectrum = Spectrum(wavelength, flux, ivar)
    # Filter on the AND mask
    spectrum.ivar[and_mask > 0] = 0

    return spectrum

