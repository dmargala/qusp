"""
Provides support for working with BOSS wavelengths

Examples
--------

Add sky lines to quasar spectrum plot:

>>> qusp.wavelength.draw_lines(qusp.wavelength.load_wavelengths('sky'))

Get a combined spectrum's fiducial pixel offset:

>>> offset = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(combined.wave[0]))

Construct a fiducial pixel wavelength array:

>>> wave = qusp.wavelength.get_fiducial_wavelength(np.arange(4800))

--------------
"""
import math
import os
import numpy as np

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
        Args:
            redshift (float): source redshift

        Returns:
            the shifted observed wavelength.
        """
        return self*(1+redshift)
    def rest(self, redshift):
        """
        Args:
            redshift (float): source redshift

        Returns:
            the shifted rest wavelength.
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

def load_wavelengths(filename, ignore_labels=False):
    """
    Loads a list of wavelengths from the specified file

    Args:
        filename (str): wavelength data filename

    Returns:
        wavelengths (list)
    """
    # Get the path that this module was loaded from.
    my_path = os.path.dirname(os.path.abspath(__file__))
    # Build the path where the filter curves should be.
    wavelegths_path = os.path.join(
        os.path.dirname(my_path), 'data', 'wavelengths')
    full_path = os.path.join(wavelegths_path, '%s.dat' % filename)
    if ignore_labels:
        wavelengths = np.genfromtxt(full_path, usecols=0)
        return [Wavelength(wave) for wave in wavelengths]
    else:
        wavelengths = np.genfromtxt(
            full_path,
            dtype={'names':('wavelengths', 'labels'), 'formats':(float, 'S100')},
            usecols=(0, 1))
        return [LabeledWavelength(*wave) for wave in wavelengths]

def get_fiducial_wavelength(pixel_index, lambda0=3500.26):
    """
    Returns the wavelength at the center of the specified index 
    of the BOSS co-add fiducial wavelength grid.

    Args:
        pixel_index (int): index of the BOSS co-add fiducial wavelength grid.

    Returns:
        wavelength (float): central wavelength of the specified index on the fiducial wavelength grid
    """
    return lambda0*(10**(1e-4*pixel_index))

def get_fiducial_pixel_index_offset(coeff0, coeff1=1e-4):
    """
    Returns the pixel index offset from the start of the
    BOSS co-add fiducial wavelength grid.

    Args:
        coeff0 (float): central wavelength (log10) of first pixel
        coeff1 (float, optional): log10 dispersion per pixel

    Returns:
        pixel index offset from the start of the fiducial wavelength grid.
    """
    if coeff1 != 1e-4:
        return 0
    delta = (math.log10(3500.26)-coeff0)/coeff1
    offset = int(math.floor(delta+0.5))
    if math.fabs(delta-offset) > 0.01:
        return 0
    return -offset

def draw_lines(waves, offset=0, delta=.1, **kwargs):
    """
    Draws vertical lines on the current plot.
    """

    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms

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

if __name__ == '__main__':
    # Tests for wavelengths module
    wave1216 = Wavelength(1216)
    assert wave1216.observed(2.5) == 4256, 'observed wavelength error'

    wave5472 = Wavelength(5472)
    assert wave5472.rest(3.5) == 1216, 'rest wavelength error'

    lambda0 = get_fiducial_wavelength(0)
    assert lambda0 == 3500.26, 'fiducial wavelength origin error'

    lambda101 = get_fiducial_wavelength(101)
    offset101 = get_fiducial_pixel_index_offset(math.log10(lambda101))
    assert offset101 == 101, 'fiducial pixel index offset error'

    waves = load_wavelengths('balmer')
    for wave in waves:
        print waves.label

