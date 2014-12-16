#!/usr/bin/env python
"""
Stacks boss spectra.
"""

import argparse
import numpy as np
import h5py

import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="more verbose output")
    parser.add_argument("--norm-min", type=float, default=1275,
        help="min norm wavelength")
    parser.add_argument("--norm-max", type=float, default=1285,
        help="max norm wavelength")
    parser.add_argument("--z-col", type=int, default=1,
        help="redshift column index")
    parser.add_argument("--tpcorr", type=str, default=None,
        help="throughput correction filename")
    parser.add_argument("-o", "--output", type=str, default=None,
        help="output file name")
    qusp.target.add_args(parser)
    qusp.Paths.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    # read target list
    target_list = qusp.target.load_target_list_from_args(args, 
        fields=[('z', float, args.z_col)])

    if args.tpcorr:
        import scipy.interpolate
        tpcorr = h5py.File(args.tpcorr)
        tpcorr_wave = tpcorr['wave'].value
    else:
        tpcorr = None

    norm_min = qusp.wavelength.Wavelength(args.norm_min)
    norm_max = qusp.wavelength.Wavelength(args.norm_max)

    # initialize stack arrays
    npixels_fiducial = 4800
    wavelength = qusp.wavelength.get_fiducial_wavelength(np.arange(npixels_fiducial))

    flux_wsum = np.zeros(npixels_fiducial)
    weight_sum = np.zeros_like(flux_wsum)

    ntargets = 0

    # loop over targets
    for target, combined in qusp.target.get_combined_spectra(target_list, tpcorr=tpcorr, paths=paths):

        norm = combined.mean_flux(norm_min.observed(target['z']), norm_max.observed(target['z']))

        if norm <= 0:
            continue

        ntargets += 1

        offset = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(combined.wavelength[0]))

        indices = slice(offset, offset+combined.npixels)

        flux_wsum[indices] += combined.ivar.values*combined.flux.values/norm
        weight_sum[indices] += combined.ivar.values

    flux_wmean = np.empty_like(flux_wsum)
    nonzero_weights = np.nonzero(weight_sum)
    flux_wmean[nonzero_weights] = flux_wsum[nonzero_weights]/weight_sum[nonzero_weights]

    if args.output:
        outfilename = args.output+'.hdf5'
        if args.verbose:
            print 'Saving stack to file: %s' % outfilename
        # save target list with sn column
        outfile = h5py.File(outfilename, 'w')

        outfile.create_dataset('flux_wmean', data=flux_wmean)
        outfile.create_dataset('weight_sum', data=weight_sum)
        outfile.create_dataset('wavelength', data=wavelength)
        outfile.attrs['ntargets'] = ntargets


if __name__ == '__main__':
    main()
