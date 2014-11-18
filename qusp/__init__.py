"""
Package for working with quasar spectra
"""

from spectrum import WavelengthFunction, SpectralFluxDensity, Spectrum, read_combined_spectrum
from target import Target
import wavelength

from model import ContinuumModel

from paths import Paths

from continuum import MeanFluxContinuum, LinearFitContinuum
