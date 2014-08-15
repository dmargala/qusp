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