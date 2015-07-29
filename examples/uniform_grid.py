#!/usr/bin/env python
import argparse

import numpy as np
import numpy.ma as ma
import h5py

import bossdata.path
import bossdata.remote
import bossdata.spec

import qusp

import fitsio

class LiteSpecFile(object):
    def __init__(self, path):
        self.hdulist = fitsio.FITS(path, mode=fitsio.READONLY)
        self.header = self.hdulist[0].read_header()

    def get_data(self, pixel_quality_mask=None):
        # Look up the HDU for this spectrum and its pixel quality bitmap.
        hdu = self.hdulist[1]
        pixel_bits = hdu['and_mask'][:]
        num_pixels = len(pixel_bits)

        # Apply the pixel quality mask, if any.
        if pixel_quality_mask is not None:
            clear_allowed = np.bitwise_not(np.uint32(pixel_quality_mask))
            pixel_bits = np.bitwise_and(pixel_bits, clear_allowed)

        # Identify the pixels with valid data.
        ivar = hdu['ivar'][:]
        bad_pixels = (pixel_bits != 0) | (ivar <= 0.0)
        good_pixels = ~bad_pixels

        # Create and fill the unmasked structured array of data.
        dtype = [('loglam', np.float32), ('flux', np.float32), ('ivar', np.float32)]
        data = np.empty(num_pixels, dtype=dtype)
        data['loglam'][:] = hdu['loglam'][:]
        data['flux'][:] = hdu['flux'][:]
        data['ivar'][:] = ivar[:]

        return ma.MaskedArray(data, mask=bad_pixels)

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    ## targets to fit
    parser.add_argument("-i", "--input", type=str, default=None,
        help="target list")
    parser.add_argument("-n", "--ntargets", type=int, default=0,
        help="number of targets to use, 0 for all")
    parser.add_argument("--z-col", type=int, default=3,
        help="column index of redshift values in input target list")
    parser.add_argument("-o", "--output", type=str, default=None,
        help="output file name")
    parser.add_argument("--forest-lo", type=float, default=1040,
        help="min forest wavelength")
    parser.add_argument("--forest-hi", type=float, default=1200,
        help="max forest wavelength")
    args = parser.parse_args()

    try:
        finder = bossdata.path.Finder()
        mirror = bossdata.remote.Manager()
    except ValueError as e:
        print(e)
        return -1

    # read target data
    fields = [('z', float, args.z_col)]
    targets = qusp.target.load_target_list(
        args.input, fields, verbose=args.verbose)

    # use the first n targets or a random sample
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)
    targets = targets[:ntargets]

    skim_redshift = np.empty(ntargets)
    for i, target in enumerate(targets):
        skim_redshift[i] = target['z']

    log_forest_max_obs = np.log10(args.forest_hi*(1+np.max(skim_redshift)))

    max_index = qusp.wavelength.get_fiducial_pixel_index_offset(log_forest_max_obs)
    max_index = np.ceil(max_index).astype(int)

    skim_flux = ma.masked_all((ntargets, max_index))
    skim_ivar = ma.masked_all((ntargets, max_index))
    skim_norm = np.zeros(ntargets)

    dtype = [('ra', np.float32), ('dec', np.float32), ('plate', np.int32), ('mjd', np.int32), ('fiber', np.int32), ('thing_id', np.int64)]
    skim_meta = np.empty(ntargets, dtype=dtype)

    print skim_flux.shape

    for i, target in enumerate(targets):
        remote_path = finder.get_spec_path(plate=target['plate'], mjd=target['mjd'], fiber=target['fiber'], lite=True)
        local_path = mirror.get(remote_path)
        spec = LiteSpecFile(local_path)

        skim_meta['ra'][i] = np.radians(np.asscalar(spec.hdulist[2]['RA'][:]))
        skim_meta['dec'][i] = np.radians(np.asscalar(spec.hdulist[2]['DEC'][:]))
        skim_meta['plate'][i] = target['plate']
        skim_meta['mjd'][i] = target['mjd']
        skim_meta['fiber'][i] = target['fiber']
        skim_meta['thing_id'][i] = np.asscalar(spec.hdulist[2]['THING_ID'][:])

        data = spec.get_data()
        loglam,flux,ivar = data['loglam'][:],data['flux'][:],data['ivar'][:]
        good_pixels = ~ivar.mask

        z = skim_redshift[i]
        log_forest_lo = max(loglam.data[0], np.log10(args.forest_lo * (1.0 + z)))
        log_forest_hi = min(loglam.data[-1], np.log10(args.forest_hi * (1.0 + z)))

        if log_forest_lo > log_forest_hi:
            print 'no good pixels in forest'
            continue

        forest_lo_index = qusp.wavelength.get_fiducial_pixel_index_offset(log_forest_lo)
        forest_lo_index = np.round(forest_lo_index).astype(int)
        forest_hi_index = qusp.wavelength.get_fiducial_pixel_index_offset(log_forest_hi)
        forest_hi_index = np.round(forest_hi_index).astype(int)

        uniform_slice = slice(forest_lo_index, forest_hi_index)

        offset = qusp.wavelength.get_fiducial_pixel_index_offset(loglam.data[0])
        spec_slice = slice(forest_lo_index-offset, forest_hi_index-offset)

        skim_flux[i, uniform_slice] = flux[spec_slice]
        skim_ivar[i, uniform_slice] = ivar[spec_slice]

        mean_lo = 1275.0 * (1.0 + z)
        mean_lo_index = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(mean_lo))
        mean_lo_index = np.round(mean_lo_index).astype(int)
        mean_hi = 1285.0 * (1.0 + z)
        mean_hi_index = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(mean_hi))
        mean_hi_index = np.round(mean_hi_index).astype(int)

        mean_slice = slice(mean_lo_index-offset, mean_hi_index-offset)

        if np.sum(ivar[mean_slice].data) <= 0:
            print 'no good pixels in norm window'
            continue

        norm, norm_weight = ma.average(flux[mean_slice].data, weights=ivar[mean_slice].data, returned=True)

        skim_norm[i] = norm

    assert np.all(skim_flux.mask == skim_ivar.mask)

    outfile = h5py.File(args.output, 'w')

    outfile.create_dataset('flux', data=skim_flux.data, compression="gzip")
    outfile.create_dataset('ivar', data=skim_ivar.data, compression="gzip")
    outfile.create_dataset('mask', data=skim_ivar.mask, compression="gzip")

    skim_loglam = np.log10(qusp.wavelength.get_fiducial_wavelength(np.arange(max_index)))
    outfile.create_dataset('loglam', data=skim_loglam, compression="gzip")

    outfile.create_dataset('z', data=skim_redshift, compression="gzip")
    outfile.create_dataset('norm', data=skim_norm, compression="gzip")

    outfile.create_dataset('meta', data=skim_meta, compression="gzip")

    outfile.close()


if __name__ == '__main__':
    main()
