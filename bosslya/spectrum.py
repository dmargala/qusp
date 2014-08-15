import numpy as np

class Spectrum:
    def __init__(self, wavelength, flux, ivar):
        self.wavelength = wavelength
        self.flux = flux
        self.ivar = ivar
        self.nPixels = len(wavelength)
        self.nZeroIvarPixels = np.sum(ivar == 0)
        self.nMaskedPixels = 0

    def findPixel(self, wavelength):
        if wavelength <= self.wavelength[0]:
            return -1
        if wavelength >= self.wavelength[-1]:
            return self.nPixels-1
        return np.argmax(self.wavelength >= wavelength)

    def getMeanFlux(self, minWavelength, maxWavelength, ivarWeighting=True):
        minPixel = self.findPixel(minWavelength)+1
        maxPixel = self.findPixel(maxWavelength)
        if minPixel > maxPixel:
            return 0
        s = slice(minPixel,maxPixel)
        weights = self.ivar[s]
        nonzero = np.nonzero(weights)
        weights = weights[nonzero] if ivarWeighting else np.ones(len(nonzero))
        wsum = np.sum(weights)
        if wsum <= 0:
            return 0
        wfluxsum = np.sum(weights*self.flux[s][nonzero])
        return wfluxsum/wsum

    def getMedianSignalToNoise(self, minWavelength, maxWavelength):
        minPixel = self.findPixel(minWavelength)+1
        maxPixel = self.findPixel(maxWavelength)
        if minPixel > maxPixel:
            return 0
        s = slice(minPixel,maxPixel)
        sn = np.fabs(self.flux[s])*np.sqrt(self.ivar[s])
        return np.median(sn[sn.nonzero()])

def readCombinedSpectrum(spPlate, fiber):
        index = fiber - 1

        coeff0 = spPlate[0].header['COEFF0']
        coeff1 = spPlate[0].header['COEFF1']

        flux = spPlate[0].data[index]
        nPixels = len(flux)
        ivar = spPlate[1].data[index]
        andMask = spPlate[2].data[index]
        orMask = spPlate[3].data[index]
        pixelDispersion = spPlate[4].data[index]
        # Calculate the wavelength sequence to use.
        wavelength = np.power(10,coeff0 + coeff1*np.arange(0, nPixels))

        # Build the spectrum (no masking yet).
        spectrum = Spectrum(wavelength, flux, ivar)
        # Filter on the AND mask?
        spectrum.ivar[andMask > 0] = 0
        # Filter on the OR mask?
        return spectrum