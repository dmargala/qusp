"""
Provides support for working with BOSS spectra.
"""
import numpy as np
from wavelength import *

import qusp

class Spectrum:
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
        self.nPixels = len(wavelength)
        self.nZeroIvarPixels = np.sum(ivar == 0)

    def findPixel(self, wavelength):
        """
        Returns the corresponding pixel index of the specified wavelength

        Args:
            wavelength (float): value

        Returns:
            pixelIndex (int): pixel index
        """
        if wavelength <= self.wavelength[0]:
            return -1
        if wavelength >= self.wavelength[-1]:
            return self.nPixels-1
        # find first pixel with a central wavelength greater than wavelength
        candidate = np.argmax(self.wavelength >= wavelength)
        # compare wavelength with this pixel's lower boundary
        if wavelength > getFiducialWavelength(candidate-0.5):
            return candidate
        else:
            return candidate - 1

    def getMeanFlux(self, minWavelength, maxWavelength, ivarWeighting=True):
        """
        Returns the mean flux between the specified wavelengths. Use ivarWeighting=False
        option to turn ignore weights.

        Args:
            minWavelength (float): minimum wavelength for mean flux calculation range.
            maxWavelength (float): maximum wavelength for mean flux calculation range.
            ivarWeighting (bool, optional): Whether or not to weight calculation using inverse variance.

        Returns:
            meanFlux (float): the mean flux between `minWavelength` and `maxWavelength`.
        """
        minPixel = self.findPixel(minWavelength)+1
        maxPixel = self.findPixel(maxWavelength)
        if minPixel > maxPixel:
            return 0
        s = slice(minPixel,maxPixel+1)
        # only use "good" pixels
        nonzero = np.nonzero(self.ivar[s])
        # use weights?
        weights = self.ivar[s][nonzero] if ivarWeighting else np.ones(len(nonzero))
        # calculate mean
        wsum = np.sum(weights)
        if wsum <= 0:
            return 0
        wfluxsum = np.sum(weights*self.flux[s][nonzero])
        return wfluxsum/wsum

    def getMedianSignalToNoise(self, minWavelength, maxWavelength):
        """
        Returns the median signal to noise ratio between the specified wavelengths.

        Args:
            minWavelength (float): minimum wavelength for median flux calculation range.
            maxWavelength (float): maximum wavelength for median flux calculation range.

        Returns:
            median (float): the median flux between `minWavelength` and `maxWavelength`.
        """
        minPixel = self.findPixel(minWavelength)+1
        maxPixel = self.findPixel(maxWavelength)
        if minPixel > maxPixel:
            return 0
        s = slice(minPixel,maxPixel+1)
        # only use "good" pixels
        nonzero = np.nonzero(self.ivar[s])
        if len(nonzero) == 0:
            return 0
        # calculate signal to noise
        sn = np.fabs(self.flux[s][nonzero])*np.sqrt(self.ivar[s][nonzero])
        # return median
        return np.median(sn)

def readCombinedSpectrum(spPlate, fiber):
    """
    Returns the combined spectrum of the specified fiber from the provided spPlate.

    Args:
        spPlate (astropy.io.fits.HDUList): spPlate file
        fiber (int,:class:`qusp.target.Target`): boss target's fiberid, or a :class:`qusp.target.Target` object.

    Returns:
        spectrum (:class:`Spectrum`): a :class:`Spectrum` object of `fiber` of `spPlate`.
    """
    # those pesky fiber numbers start at 1 but the fits table is 0-indexed
    if type(fiber) is qusp.target.Target:
        index = fiber['fiber'] - 1
    else:
        index = fiber - 1

    coeff0 = spPlate[0].header['COEFF0']
    coeff1 = spPlate[0].header['COEFF1']

    flux = spPlate[0].data[index]
    ivar = spPlate[1].data[index]
    andMask = spPlate[2].data[index]
    orMask = spPlate[3].data[index]
    pixelDispersion = spPlate[4].data[index]
    # Calculate the wavelength sequence to use.
    nPixels = len(flux)
    wavelength = np.power(10,coeff0 + coeff1*np.arange(0, nPixels))

    # Build the spectrum (no masking yet).
    spectrum = Spectrum(wavelength, flux, ivar)
    # Filter on the AND mask
    spectrum.ivar[andMask > 0] = 0

    return spectrum


def plotSpectrum(spectrum, **kwargs):
    if isinstance(spectrum, basestring):
        plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
        fullName = os.path.join(fitsPath,str(target.plate),plateFileName)
        spPlate = fits.open(fullName)
        combined = qusp.readCombinedSpectrum(spPlate, target.fiber)
        spPlate.close()
    elif isinstance(spectrum, Spectrum):
        combined = spectrum
    else:
        assert False, 'Invalid spectrum argument'
    x = combined.wavelength
    y = combined.flux
    plt.plot(x, y, **kwargs)
    plt.xlim([x[0], x[-1]])
    plt.ylabel(r'Flux $(10^{-17} erg/cm^2/s/\AA)$')
