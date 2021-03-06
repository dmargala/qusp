#!/usr/bin/env python
import argparse

import numpy as np
import numpy.ma as ma
import h5py

import bossdata.path
import bossdata.remote
import bossdata.spec
import bossdata.meta

import qusp

import fitsio

from progressbar import ProgressBar, Percentage, Bar

class FullSpecFile(object):
    def __init__(self, path):
        self.hdulist = fitsio.FITS(path, mode=fitsio.READONLY)
        self.header = self.hdulist[0].read_header()

    def get_data(self):
        # Look up the HDU for this spectrum and its pixel quality bitmap.
        hdu = self.hdulist[1]
        pixel_bits = hdu['and_mask'][:]
        num_pixels = len(pixel_bits)

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
    parser.add_argument("-o", "--output", type=str, default=None,
        help="output file name")
    parser.add_argument("--forest-lo", type=float, default=1040,
        help="min forest wavelength")
    parser.add_argument("--forest-hi", type=float, default=1200,
        help="max forest wavelength")
    parser.add_argument("--norm-lo", type=float, default=1275,
        help="min forest wavelength")
    parser.add_argument("--norm-hi", type=float, default=1285,
        help="max forest wavelength")
    parser.add_argument("--spall-redshift", action="store_true",
        help="Use redshift from hdu 2 of spec file, instead of quasar catalog redshift")
    parser.add_argument("--max-fid-index", type=int, default=1900,
        help="Maximum fiducial pixel index")
    parser.add_argument("--mock", action="store_true",
        help="Mock input data")
    args = parser.parse_args()

    try:
        finder = bossdata.path.Finder()
        mirror = bossdata.remote.Manager()
    except ValueError as e:
        print(e)
        return -1

    # read target data
    targets = qusp.target.load_target_list(args.input, verbose=args.verbose)

    # use the first n targets or a random sample
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)
    targets = targets[:ntargets]

    # determine maximum forest wavelength observed
    skim_redshift = np.empty(ntargets)
    if not args.spall_redshift:
        quasar_catalog = bossdata.meta.Database(quasar_catalog=True, lite=False)

    #log_forest_max_obs = np.log10(args.forest_hi*(1+np.max(skim_redshift)))
    # convert max observed wavelength to fiducial pixel index
    #max_index = qusp.wavelength.get_fiducial_pixel_index_offset(log_forest_max_obs)
    max_index = args.max_fid_index
    #max_index = np.ceil(max_index).astype(int)

    # arrays for skim data
    skim_flux = ma.masked_all((ntargets, max_index))
    skim_ivar = ma.masked_all((ntargets, max_index))
    skim_norm = np.zeros(ntargets)

    # target meta data
    dtype = [('ra', np.float32), ('dec', np.float32), ('plate', np.int32), ('mjd', np.int32), ('fiber', np.int32), ('thing_id', np.int64)]
    skim_meta = np.empty(ntargets, dtype=dtype)

    if args.verbose:
        progress_bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=ntargets).start()

    bad_forest = []
    bad_norm = []
    bad_target = []

    for i, target in enumerate(targets):
        if args.verbose:
            progress_bar.update(i)

        # get lite spec filename and load data
        remote_path = finder.get_spec_path(
            plate=target['plate'], mjd=target['mjd'], fiber=target['fiber'],
            lite=False, compressed=args.mock, mock=args.mock)
        local_path = mirror.get(remote_path, progress_min_size=0.1)
        spec = FullSpecFile(local_path)

        # copy meta data
        skim_meta['ra'][i] = np.radians(np.asscalar(spec.hdulist[2]['RA'][:]))
        skim_meta['dec'][i] = np.radians(np.asscalar(spec.hdulist[2]['DEC'][:]))
        skim_meta['plate'][i] = target['plate']
        skim_meta['mjd'][i] = target['mjd']
        skim_meta['fiber'][i] = target['fiber']
        skim_meta['thing_id'][i] = np.asscalar(spec.hdulist[2]['THING_ID'][:])

        # look up redshift
        if args.spall_redshift:
            z = np.asscalar(spec.hdulist[2]['Z'][:])
        else:
            where = 'PLATE=%s and MJD=%s and FIBER=%s' % (target['plate'], target['mjd'], target['fiber'])
            result = quasar_catalog.select_all(where=where, what='Z_VI', max_rows=1)
            if result is None:
                bad_target.append(i)
                #print '{}: not in quasar catalog'.format(target['target'])
                continue
            z = result['Z_VI'][0]
        skim_redshift[i] = z

        # process spectrum data
        data = spec.get_data()
        loglam,flux,ivar = data['loglam'][:],data['flux'][:],data['ivar'][:]
        # determine fiducial wavelength offsets of forest data
        log_forest_lo = max(loglam.data[0], np.log10(args.forest_lo * (1.0 + z)))
        forest_lo_index = qusp.wavelength.get_fiducial_pixel_index_offset(log_forest_lo)
        forest_lo_index = np.round(forest_lo_index).astype(int)
        log_forest_hi = min(loglam.data[-1], np.log10(args.forest_hi * (1.0 + z)))
        forest_hi_index = qusp.wavelength.get_fiducial_pixel_index_offset(log_forest_hi)
        forest_hi_index = np.round(forest_hi_index).astype(int)
        # check for valid forest window
        if forest_lo_index > forest_hi_index:
            bad_forest.append(i)
            #print '{}: no forest pixels [{}:{}], z = {}'.format(target['target'],
            #    np.power(10.0, log_forest_lo), np.power(10.0, log_forest_hi), z)
            continue
        uniform_slice = slice(forest_lo_index, forest_hi_index)
        offset = qusp.wavelength.get_fiducial_pixel_index_offset(loglam.data[0])
        spec_slice = slice(forest_lo_index-offset, forest_hi_index-offset)
        # check for valid weights in forest window
        if ma.sum(ivar[spec_slice].mask) == len(ivar[spec_slice]):
            bad_forest.append(i)
            #print '{}: no unmasked pixels in forest [{}:{}], z = {}'.format(target['target'],
            #    np.power(10.0, log_forest_lo), np.power(10.0, log_forest_hi), z)
            continue
        # find normalization window
        norm_lo = args.norm_lo * (1.0 + z)
        norm_lo_index = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(norm_lo))
        norm_lo_index = np.round(norm_lo_index).astype(int)
        norm_hi = args.norm_hi * (1.0 + z)
        norm_hi_index = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(norm_hi))
        norm_hi_index = np.round(norm_hi_index).astype(int)
        norm_slice = slice(norm_lo_index-offset, norm_hi_index-offset)
        # check for valid weights in normalization window
        if np.sum(ivar[norm_slice].data) <= 0:
            bad_norm.append(i)
            #print '{}: no good pixels in norm window [{}:{}], z = {}'.format(target['target'], norm_lo, norm_hi, z)
            continue
        # calculate normalization as ivar weighted mean flux
        norm, norm_weight = ma.average(flux[norm_slice].data, weights=ivar[norm_slice].data, returned=True)
        # verify normalization is valid
        if norm <= 0:
            bad_norm.append(i)
            #print '{}: norm <= 0'.format(target['target'])
            continue

        # save normalization
        skim_norm[i] = norm
        # copy forest data to skim
        skim_flux[i, uniform_slice] = flux[spec_slice]
        skim_ivar[i, uniform_slice] = ivar[spec_slice]

    if args.verbose:
        progress_bar.finish()

    print 'Number of spectra with bad forest window: %d' % len(bad_forest)
    print 'Number of spectra with bad norm: %d' % len(bad_norm)
    print 'Number of spectra not in quasar catalog: %d' % len(bad_target)

    # verify flux and ivar masks are equal
    assert np.all(skim_flux.mask == skim_ivar.mask)

    masked_rows = (np.sum(skim_ivar.mask, axis=1) == max_index)
    save_rows = ~masked_rows
    print 'Saving %d rows...' % np.sum(save_rows)

    outfile = h5py.File(args.output, 'w')
    # save args
    outfile.attrs['forest_lo'] = args.forest_lo
    outfile.attrs['forest_hi'] = args.forest_hi
    outfile.attrs['norm_lo'] = args.norm_lo
    outfile.attrs['norm_hi'] = args.norm_hi
    # save the index of the maximum observed forest wavelength
    outfile.attrs['max_fid_index'] = max_index
    # save uniform wavelength grid
    skim_loglam = np.log10(qusp.wavelength.get_fiducial_wavelength(np.arange(max_index)))
    outfile.create_dataset('loglam', data=skim_loglam, compression="gzip")
    # save pixel flux, ivar, and mask
    outfile.create_dataset('flux', data=skim_flux.data[save_rows], compression="gzip")
    outfile.create_dataset('ivar', data=skim_ivar.data[save_rows], compression="gzip")
    outfile.create_dataset('mask', data=skim_ivar.mask[save_rows], compression="gzip")
    # save redshifts from input target list
    outfile.create_dataset('z', data=skim_redshift[save_rows], compression="gzip")
    # save additional quantities
    outfile.create_dataset('norm', data=skim_norm[save_rows], compression="gzip")
    # save target meta data
    outfile.create_dataset('meta', data=skim_meta[save_rows], compression="gzip")
    # save masked target meta data
    outfile.create_dataset('masked_meta', data=skim_meta[masked_rows], compression="gzip")

    outfile.attrs['coeff0'] = skim_loglam[0]
    outfile.attrs['coeff1'] = 1e-4
    outfile.attrs['wave_min'] = 0

    outfile.close()


if __name__ == '__main__':
    main()
