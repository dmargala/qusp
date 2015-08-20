#!/usr/bin/env python
"""
"""

import argparse
import os

import h5py
import healpy as hp
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits

import qusp

import glob

from progressbar import ProgressBar, Percentage, Bar

class DeltaLOS():
    """
    Represents a delta file (delta field along a line of sight)
    """
    def __init__(self, filename):
        hdulist = fits.open(filename)
        data = hdulist[1].data
        self.loglam = data['loglam']
        self.delta = data['delta']
        self.weight = data['weight']
        self.r_comov = data['r_comov']
        self.cont = data['cont']
        self.msha = data['msha']
        self.mabs = data['mabs']
        self.ivar = data['ivar']

        self.flux = (1+self.delta)*self.cont*self.msha*self.mabs

        header = hdulist[1].header
        self.z = header['z']
        self.plate = header['plate']
        self.mjd = header['mjd']
        self.fiber = header['fiberid']
        self.thing_id = header['thing_id']
        self.ra = header['ra']
        self.dec = header['dec']
        self.p0 = header['p0']
        self.p1 = header['p1']

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', type=str, default=None,
        help='output file name')
    parser.add_argument('--input', type=str, default=None,
        help='input filename pattern')
    args = parser.parse_args()

    delta_filenames = glob.glob(args.input)

    print 'Found %d filenames matching pattern "%s"' % (len(delta_filenames), args.input)

    outfile = h5py.File(args.output, 'w')
    lines_of_sight = outfile.create_group('lines_of_sight')

    num_sightlines = len(delta_filenames)

    # loop over targets
    progress_bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=num_sightlines).start()
    for i, filename in enumerate(delta_filenames):
        progress_bar.update(i)

        delta = DeltaLOS(filename)

        # save to hdf5 file
        los = lines_of_sight.create_group('%s' % delta.thing_id)
        los.attrs['plate'] = delta.plate
        los.attrs['mjd'] = delta.mjd
        los.attrs['fiber'] = delta.fiber
        los.attrs['ra'] = delta.ra
        los.attrs['dec'] = delta.dec
        los.attrs['z'] = delta.z
        los.attrs['p0'] = delta.p0
        los.attrs['p1'] = delta.p1
        los.create_dataset('loglam', data=delta.loglam, dtype='f4')
        # los.create_dataset('flux', data=delta.flux, dtype='f4')
        los.create_dataset('ivar', data=delta.ivar, dtype='f4')
        los.create_dataset('delta', data=delta.delta, dtype='f8')
        los.create_dataset('weight', data=delta.weight, dtype='f8')
        los.create_dataset('r_comov', data=delta.r_comov, dtype='f4')

    outfile.close()
    progress_bar.finish()


if __name__ == '__main__':
    main()
