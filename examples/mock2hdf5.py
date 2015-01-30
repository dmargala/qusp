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

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action='store_true',
        help='more verbose output')
    parser.add_argument('--input', type=str, default=None,
        help='input file name')
    parser.add_argument('--output', type=str, default=None,
        help='output file name')
    parser.add_argument('--mock-version', type=str, default='M3_0_0',
        help='version of mocks')
    qusp.Paths.add_args(parser)
    qusp.target.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    # read target list
    target_list = qusp.target.load_target_list_from_args(args)


    outfile = h5py.File(args.output, 'w')

    outfile_delta = outfile.create_group('delta_field')

    mock_root = os.path.join(paths.boss_root, args.mock_version, 'rawlite')

    wave_lya = 1216.

    n_failed = 0
    # loop over targets
    for i, target in enumerate(target_list):
        if i % 10000 == 0:
            print i

        plate, mjd, fiber = (target['plate'], target['mjd'], target['fiber'])
        mock_name = os.path.join(mock_root, str(plate), 'mockrawShort-%04d-%5d-%04d.fits' % (plate, mjd, fiber))

        try:
            hdulist = fits.open(mock_name)
        except IOError:
            n_failed += 1
            print 'Failed trying to open: %s (%d/%d)' % (mock_name, n_failed, i)
            continue

        z = hdulist[0].header['m_z']
        ra = hdulist[0].header['m_ra']
        dec = hdulist[0].header['m_dec']

        coeff0 = hdulist[0].header['coeff0']
        coeff1 = hdulist[0].header['coeff1']

        mock_f = hdulist[2].data.field(0)
        mock_ivar = np.ones_like(mock_f)

        mock_lambda = np.power(10, coeff0 + coeff1*np.arange(len(mock_f)))
        absorber_z = mock_lambda/wave_lya - 1

        # save delta field along this line of sight
        outfile_delta_target = outfile_delta.create_group(target.to_string())
        outfile_delta_target.attrs['ra'] = ra
        outfile_delta_target.attrs['dec'] = dec
        outfile_delta_target.attrs['z'] = z
        outfile_delta_target.create_dataset('absorber_z', data=absorber_z, dtype='f4')
        outfile_delta_target.create_dataset('absorber_delta', data=mock_f, dtype='f4')
        outfile_delta_target.create_dataset('absorber_ivar', data=mock_ivar, dtype='f4')


if __name__ == '__main__':
    main()