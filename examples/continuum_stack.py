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
    parser.add_argument("--unweighted", action="store_true",
        help="unweighted stack")
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

    min_fid_index = -6150
    max_fid_index = -670
    fid_npixels = max_fid_index - min_fid_index

    continuum_wave_centers = qusp.wavelength.get_fiducial_wavelength(np.arange(min_fid_index, max_fid_index))

    print qusp.wavelength.get_fiducial_wavelength(min_fid_index), qusp.wavelength.get_fiducial_wavelength(max_fid_index-1)
    print fid_npixels

    redshifted_fluxes = np.ma.empty((len(target_list), fid_npixels))
    redshifted_fluxes[:] = np.ma.masked

    # loop over targets
    for target, combined in qusp.target.get_combined_spectra(target_list, tpcorr=tpcorr, paths=paths):

        continuum_wave = combined.wavelength/(1+target['z'])

        fid_offsets = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(continuum_wave))
        fid_offset_indices = np.round(fid_offsets).astype(int)
        continuum_indices = fid_offset_indices - min_fid_index

        valid_pixels = (continuum_indices < fid_npixels) & (continuum_indices >= 0)

        if norm_min.observed(target['z']) < combined.wavelength[0]:
            print 'norm region not in range'
            continue
        norm = combined.mean_flux(norm_min.observed(target['z']), norm_max.observed(target['z']))

        if norm <= 0:
            print 'norm less than 0', target.to_string(), target['z'] 
            continue

        redshifted_fluxes[ntargets, continuum_indices[valid_pixels]] = combined.flux.values[valid_pixels]/norm

        ntargets += 1


    print redshifted_fluxes.shape

    median_flux = np.ma.median(redshifted_fluxes, axis=0)
    mean_flux = np.ma.mean(redshifted_fluxes, axis=0)

    if args.output:
        outfilename = args.output+'.hdf5'
        if args.verbose:
            print 'Saving stack to file: %s' % outfilename
        # save target list with sn column
        outfile = h5py.File(outfilename, 'w')

        outfile.create_dataset('median_flux', data=median_flux)
        outfile.create_dataset('mean_flux', data=mean_flux)
        outfile.create_dataset('wavelength', data=continuum_wave_centers)
        outfile.attrs['ntargets'] = ntargets


if __name__ == '__main__':
    main()
