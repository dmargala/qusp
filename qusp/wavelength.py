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

class LabeledWavelength(Wavelength):
    def __init__(self, value, label):
        Wavelength.__init__(value)
        self.label = label

# SDSS lines
# http://classic.sdss.org/dr7/algorithms/linestable.html

SkyLineList = [3933.68, 3968.47, 5578.5,5894.6,6301.7,7246.0]
SkyLabels = ['Ca II','Ca II','sky','sky','sky','sky']
SkyLines = [Wavelength(value) for value in SkyLineList]

BallmerLines = [3646,3835,3889,3970,4102,4341,4861,6563]
BallmerLabels = [r'$Hlimit$',r'$H\eta$',r'$H\zeta$',r'$H\epsilon$',r'$H\delta$',r'$H\gamma$',r'$H\beta$',r'$H\alpha$']

QuasarEmissionLines = [912, 972, 1026,1033.82,1215.6701,1225,1240.81,1305.53,
    1335.31,1397.61,1399.8,1549.48,1640.4,1665.85,1857.4,
    1908.734,2326.0,2439.5,2799.117,3346.79,3426.85,3727.092,
    3729.875,3889.0,4072.3,4102.89,4341.68,4364.436,4862.68,
    4932.603,4960.295,5008.240,6302.046,6365.536,6529.03,
    6549.86,6564.61,6585.27,6718.29,6732.67]

QuasarEmissionLabels = ['LyLimit',r'Ly$\gamma$',r'Ly$\beta$','O VI',r'Ly$\alpha$','Si III','N V','O I','C II',
    'Si IV','O IV','C IV','He II','O III','Al III',
    'C III','C II','Ne IV','Mg II','Ne V','Ne VI','O II','O II',
    'He I','S II',r'H$\delta$',r'H$\gamma$','O III',r'H$\beta$','O III','O III','O III',
    'O I','O I','N I','N II',r'H$\alpha$','N II','S II','S II']

QuasarLines = [Wavelength(value) for value in QuasarEmissionLines]


def drawLines(waves, labels, offset=0, delta=.1, **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms

    wavemin, wavemax = plt.gca().get_xlim()

    ax = plt.gca()
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for i,(wave,label) in enumerate(zip(waves,labels)):
        if wave < wavemin or wave > wavemax:
            continue
        plt.axvline(wave,**kwargs)
        plt.text(wave, offset+(i%2)*delta, label, transform=trans, horizontalalignment='left')
