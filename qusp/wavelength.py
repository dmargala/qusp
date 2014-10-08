"""
Provides support for working with BOSS wavelengths

Examples
--------

Get a combined spectrum fiducial pixel offset:

>>> offset = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(combined.wave[0]))

Construct a fiducial pixel wavelength array:

>>> wave = qusp.wavelength.get_fiducial_wavelength(np.arange(4800))

Add sky lines to quasar spectrum plot:

>>> qusp.wavelength.draw_lines(load_wavelengths('sky'))

--------------
"""
import math
import os
import numpy as np

def get_fiducial_wavelength(pixel_index):
    """
    Returns the wavelength at the center of the specified
    index of the BOSS co-add fiducial wavelength grid.
    """
    return 3500.26*(10**(1e-4*pixel_index))

def get_fiducial_wavelength_ratio(lambda1, lambda2=3500.26):
    """
    Returns the fiducial wavelength ratio
    """
    return 1e4*math.log10(lambda1/lambda2)

def get_fiducial_pixel_index_offset(coeff0, coeff1=1e-4):
    """
    Returns the pixel index offset from the start of the
    BOSS co-add fiducial wavelength grid.
    """
    if coeff1 != 1e-4:
        return 0
    delta = (math.log10(3500.26)-coeff0)/coeff1
    offset = int(math.floor(delta+0.5))
    if math.fabs(delta-offset) > 0.01:
        return 0
    return -offset

class Wavelength(float):
    """
    A Wavelength is a float
    """
    def __init__(self, value):
        float.__init__(value)
    def __new__(cls, value, *args, **kwargs):
        return float.__new__(cls, value)
    def observed(self, redshift):
        """
        Returns the shifted observed wavelength.
        """
        return self*(1+redshift)
    def rest(self, redshift):
        """
        Returns the shifted rest wavelength.
        """
        return self/(1+redshift)

class LabeledWavelength(Wavelength):
    """
    A LabeledWavelength is a Wavelength with a label attribute
    """
    def __init__(self, value, label):
        Wavelength.__init__(self, value)
        self.label = label
    def __str__(self):
        return str((self, self.label))

def load_wavelengths(filename):
    """
    loads wavelength data

    Args:
        filename (str): wavelength data filename

    Returns:
        wavelengths
    """
    # Get the path that this module was loaded from.
    my_path = os.path.dirname(os.path.abspath(__file__))
    # Build the path where the filter curves should be.
    wavelegths_path = os.path.join(
        os.path.dirname(my_path), 'data', 'wavelengths')
    wavelengths = np.genfromtxt(
        os.path.join(wavelegths_path, '%s.dat' % filename),
        dtype={'names':('wavelengths', 'labels'), 'formats':(float, 'S10')},
        usecols=(0, 1))
    return [LabeledWavelength(*wave) for wave in wavelengths]

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

def draw_lines(waves, offset=0, delta=.1, **kwargs):
    """
    Draws vertical lines on the current plot.
    """

    wavemin, wavemax = plt.gca().get_xlim()

    transform = transforms.blended_transform_factory(
        plt.gca().transData, plt.gca().transAxes)

    for index, wave in enumerate(waves):
        if wave < wavemin or wave > wavemax:
            continue
        plt.axvline(wave, **kwargs)
        try:
            plt.text(wave, offset+(index%2)*delta, wave.label,
                     transform=transform, horizontalalignment='left')
        except AttributeError:
            pass

def main():
    """
    tests for wavelengths module
    """
    waves = load_wavelengths('sky')
    for wave in waves:
        print wave.label

if __name__ == '__main__':
    main()
