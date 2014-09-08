import math

def getFiducialWavelength(pixelIndex):
    return 3500.26*(10**(1e-4*pixelIndex))

def getFiducialWavelengthRatio(lambda1, lambda2=3500.26):
    return 1e4*math.log10(lambda1/lambda2)

def getFiducialPixelIndexOffset(coeff0, coeff1=1e-4):
    if coeff1 != 1e-4:
        return 0
    delta = (math.log10(3500.26)-coeff0)/coeff1
    offset = int(math.floor(delta+0.5))
    if math.fabs(delta-offset) > 0.01:
        return 0
    return -offset

class Wavelength(float):
    def __init__(self, value):
        float.__init__(value)
    def __new__(cls, value, *args, **kwargs):
        return float.__new__(cls, value)
    def observed(self, redshift):
        return self*(1+redshift)
    def rest(self, redshift):
        return self/(1+redshift)

SkyLineList = [5578.5,5894.6,5894.6,7246.0]
SkyLines = [Wavelength(value) for value in SkyLineList]


QuasarEmissionLines = [1033.82,1215.24,1240.81,1305.53,
    1335.31,1397.61,1399.8,1549.48,1640.4,1665.85,1857.4,
    1908.734,2326.0,2439.5,2799.117,3346.79,3426.85,3727.092,
    3729.875,3889.0,4072.3,4102.89,4341.68,4364.436,4862.68,
    4932.603,4960.295,5008.240,6302.046,6365.536,6529.03,
    6549.86,6564.61,6585.27,6718.29,6732.67]

QuasarEmissionLabels = ['O VI',r'Ly$\alpha$','N V','O I','C II',
    'Si IV','Si IV + O IV','C IV','He II','O III','Al III',
    'C III','C II','Ne IV','Mg II','Ne V','Ne VI','O II','O II',
    'He I','S II',r'H$\delta$',r'H$\gamma$','O III',r'H$\beta$','O III','O III','O III',
    'O I','O I','N I','N II',r'H$\alpha$','N II','S II','S II']

QuasarLines = [Wavelength(value) for value in QuasarEmissionLines]


