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
    ntargets = 0

    continuum_wave_min = 975
    continuum_wave_max = 3000
    continuum_npixels = 1000
    continuum_wave_delta = float(continuum_wave_max-continuum_wave_min)/(continuum_npixels)

    continuum_wave_centers=0.5*continuum_wave_delta + np.linspace(
        continuum_wave_min, continuum_wave_max, continuum_npixels, endpoint=False)
    continuum_wave_bins = np.linspace(continuum_wave_min, continuum_wave_max, continuum_npixels+1, endpoint=True)

    flux_wsum = np.zeros(continuum_npixels)
    weight_sum = np.zeros_like(flux_wsum)

    print ('Continuum model wavelength bin centers span [%.2f:%.2f] with %d bins.' %
        (continuum_wave_centers[0], continuum_wave_centers[-1], continuum_npixels))

    # loop over targets
    for target, combined in qusp.target.get_combined_spectra(target_list, tpcorr=tpcorr, paths=paths):

        continuum_wave = combined.wavelength/(1+target['z'])
        continuum_indices = np.floor((continuum_wave-continuum_wave_min)/continuum_wave_delta).astype(int)
        ivar = combined.ivar.values

        valid_pixels = (continuum_indices < len(continuum_wave_centers)) & (continuum_indices >= 0)

        npixels = np.sum(valid_pixels)

        if npixels <= 0:
            if args.verbose:
                print 'No good pixels for target %s (z=%.2f)' % (
                    target['target'], target['z'])
            continue

        if norm_min.observed(target['z']) < combined.wavelength[0]:
            continue
        norm = combined.mean_flux(norm_min.observed(target['z']), norm_max.observed(target['z']))

        if norm <= 0:
            continue

        ntargets += 1

        continuum_indices = continuum_indices[valid_pixels]
        flux = combined.flux.values[valid_pixels]/norm
        ivar = combined.ivar.values[valid_pixels]

        for i,pixel in enumerate(continuum_indices):
            weight_sum[pixel] += ivar[i]
            flux_wsum[pixel] += ivar[i]*flux[i]

        # offset = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(combined.wavelength[0]))
        # indices = slice(offset, offset+combined.npixels)
        # flux_wsum[indices] += combined.ivar.values[valid_pixels]*combined.flux.values[valid_pixels]/norm
        # weight_sum[indices] += combined.ivar.values[valid_pixels]

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
        outfile.create_dataset('wavelength', data=continuum_wave_centers)
        outfile.attrs['ntargets'] = ntargets


if __name__ == '__main__':
    main()
