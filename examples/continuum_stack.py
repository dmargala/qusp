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
    parser.add_argument("--use-lite", action="store_true",
        help="use lite spectra files")
    parser.add_argument("--keep", type=str, default=None,
        help="only keep these targets")
    parser.add_argument("--observed", action="store_true",
        help="stack in observed frame")
    qusp.target.add_args(parser)
    qusp.Paths.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    # read target list
    target_list = qusp.target.load_target_list_from_args(args, fields=[('z', float, args.z_col)])

    if args.keep:
        keep_list = set(np.loadtxt(args.keep, dtype='S15').tolist())
        print len(keep_list)
        keep_target_list = []
        for target in target_list:
            if args.use_lite:
                tstring = '%d-%d-%d' %(target['boss_plate'], target['boss_mjd'], target['boss_fiber'])
            else:
                tstring = target.to_string()
            if tstring in keep_list:
                keep_target_list.append(target)
        target_list = keep_target_list

    print len(target_list)


    if args.tpcorr:
        tpcorr = h5py.File(args.tpcorr)
        tpcorr_wave = tpcorr['wave'].value
    else:
        tpcorr = None

    norm_min = qusp.wavelength.Wavelength(args.norm_min)
    norm_max = qusp.wavelength.Wavelength(args.norm_max)

    # initialize stack arrays
    ntargets = 0

    if args.observed:
        min_fid_index = 0
        max_fid_index = 4800
    else:
        min_fid_index = -6150
        max_fid_index = -670
    fid_npixels = max_fid_index - min_fid_index

    continuum_wave_centers = qusp.wavelength.get_fiducial_wavelength(np.arange(min_fid_index, max_fid_index))

    print qusp.wavelength.get_fiducial_wavelength(min_fid_index), qusp.wavelength.get_fiducial_wavelength(max_fid_index-1)
    print fid_npixels

    norm_min_index = np.round(qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(norm_min))).astype(int)-min_fid_index
    norm_max_index = np.round(qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(norm_max))).astype(int)-min_fid_index

    redshifted_fluxes = np.ma.empty((len(target_list), fid_npixels))
    redshifted_fluxes[:] = np.ma.masked

    def get_lite_spectra(target_list):
        for target in target_list:
            try:
                spec = qusp.target.get_lite_spectrum(target, paths=paths)
                if tpcorr:
                    from scipy.interpolate import interp1d
                    wave = tpcorr['wave'].value
                    try:
                        value = tpcorr['%s/%s/%s' % (target['plate'], target['mjd'], target['fiber'])].value
                    except KeyError:
                        print 'No tpcorr entry for: %s' % target.to_string()
                        continue
                    correction = interp1d(wave, value, kind='linear', copy=False)
                    spec = spec.create_corrected(correction)
                yield target, spec
            except IOError:
                continue

    if args.use_lite:
        spectrum_gen = get_lite_spectra(target_list)
    else:
        spectrum_gen = qusp.target.get_corrected_spectrum(target_list, tpcorr=tpcorr, paths=paths)

    targets_used = []

    # loop over targets
    for target, combined in spectrum_gen:

        if args.observed:
            continuum_wave = combined.wavelength
        else:
            continuum_wave = combined.wavelength/(1+target['z'])

        fid_offsets = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(continuum_wave))
        fid_offset_indices = np.round(fid_offsets).astype(int)
        continuum_indices = fid_offset_indices - min_fid_index

        valid_pixels = (continuum_indices < fid_npixels) & (continuum_indices >= 0)
        norm_pixels = (continuum_indices < norm_max_index) & (continuum_indices >=  norm_min_index)

        normfluxes = np.ma.masked_array(combined.flux.values[norm_pixels], mask=(combined.ivar.values[norm_pixels] == 0))
        norm = np.ma.average(normfluxes, weights=combined.ivar.values[norm_pixels])

        if np.sum(normfluxes.mask) == np.sum(norm_pixels):
            print 'no unmasked pixels in norm region', target.to_string(), target['z'] 
            continue

        if norm <= 0:
            print 'norm less than 0', target.to_string(), target['z'] 
            continue

        redshifted_fluxes[ntargets, continuum_indices[valid_pixels]] = combined.flux.values[valid_pixels]/norm

        ntargets += 1
        targets_used.append(target.to_string())

    print redshifted_fluxes.shape
    print ntargets

    with open(args.output+'.txt', 'w') as outfile:
        for target in sorted(targets_used):
            outfile.write('%s\n'%target)

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
