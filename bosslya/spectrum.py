import numpy as np
from wavelength import *

class Spectrum:
    def __init__(self, wavelength, flux, ivar):
        self.wavelength = wavelength
        self.flux = flux
        self.ivar = ivar
        self.nPixels = len(wavelength)
        self.nZeroIvarPixels = np.sum(ivar == 0)

    def findPixel(self, wavelength):
        """
        Returns the corresponding pixel index of the specified wavelength
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
        # use weights?
        weights = self.ivar[s][nonzero] if ivarWeighting else np.ones(len(nonzero))
        # calculate mean
        wsum = np.sum(weights)
        wfluxsum = np.sum(weights*self.flux[s][nonzero])
        return wfluxsum/wsum

    def getMedianSignalToNoise(self, minWavelength, maxWavelength):
        """
        Returns the median signal to noise ratio between the specified wavelengths.
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
    """
    # those pesky fiber numbers start at 1
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