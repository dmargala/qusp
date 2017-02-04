#!/usr/bin/env python
import argparse

import numpy as np
import numpy.ma as ma
import h5py

import qusp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate
import fitsio

from restframe_work import export_exact_image

from progressbar import ProgressBar, Percentage, Bar

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## targets to fit
    parser.add_argument("--name", type=str, default=None,
        help="base name of combined skim file")
    parser.add_argument("--subsample-step", type=int, default=1000,
        help="step size used for subsampling observations")
    parser.add_argument("--dont-save", action="store_true",
        help="dont save delta field (just do all the preprocessing)")
    args = parser.parse_args()

    # import data
    skim = h5py.File(args.name+'.hdf5', 'r')
    norm = skim['norm'][:][:,np.newaxis]
    loglam = skim['loglam'][:]
    wave = np.power(10.0, loglam)

    quasar_redshifts = skim['z'][:]

    linear_continuum = h5py.File(args.name+'-linear-continuum.hdf5', 'r')
    params_a = linear_continuum['params_a'].value
    params_b = linear_continuum['params_b'].value
    continuum = linear_continuum['continuum'].value
    continuum_wave = linear_continuum['continuum_wave'].value
    continuum_interp = scipy.interpolate.UnivariateSpline(continuum_wave, continuum, s=0, ext=1)

    wave_lya = linear_continuum.attrs['wave_lya']
    abs_alpha = linear_continuum.attrs['abs_alpha']
    abs_beta = linear_continuum.attrs['abs_beta']
    forest_wave_ref = linear_continuum.attrs['forest_wave_ref']

    print 'Adjusting weights for pipeline variance and LSS variance...'

    forest_min_z = linear_continuum.attrs['forest_min_z']
    forest_max_z = linear_continuum.attrs['forest_max_z']
    forest_dz = 0.1
    forest_z_bins = np.arange(forest_min_z, forest_max_z + forest_dz, forest_dz)
    var_lss = scipy.interpolate.UnivariateSpline(forest_z_bins, 0.05 + 0.06*(forest_z_bins - 2.0)**2, s=0)
    var_pipe_scale = scipy.interpolate.UnivariateSpline(forest_z_bins, 0.7 + 0.2*(forest_z_bins - 2.0)**2, s=0)

    forest_pixel_redshifts = wave/wave_lya - 1
    abs_coefs = abs_alpha*np.power(1+forest_pixel_redshifts, abs_beta)

    forest_wave_refs = forest_wave_ref*(1+quasar_redshifts)
    def model_flux(a, b):
        return a*np.power(wave/forest_wave_refs[:,np.newaxis], b)*continuum_interp(wave/(1+quasar_redshifts[:,np.newaxis]))*np.exp(-abs_coefs)

    mflux = model_flux(params_a[:,np.newaxis],params_b[:,np.newaxis])

    # (1.0 + quasar_redshifts[:,np.newaxis])*forest_wave/args.wave_lya - 1.0

    print forest_pixel_redshifts.shape

    pixel_mask = skim['mask'][:]

    print pixel_mask.shape

    flux = np.ma.MaskedArray(skim['flux'][:], mask=pixel_mask)
    ivar = np.ma.MaskedArray(skim['ivar'][:], mask=pixel_mask)

    delta_flux = flux/mflux - 1.0
    delta_ivar = ivar*mflux*mflux

    delta_weight = delta_ivar*var_pipe_scale(forest_pixel_redshifts)
    delta_weight = delta_weight/(1 + delta_weight*var_lss(forest_pixel_redshifts))

    redshift_order = np.argsort(quasar_redshifts)
    export_exact_image(args.name+'-delta-flux.png', delta_flux[redshift_order][::args.subsample_step], dpi=100,
        vmin=-5, vmax=5, cmap=plt.get_cmap('bwr'), origin='lower')
    export_exact_image(args.name+'-delta-weight.png', ma.log10(delta_flux[redshift_order][::args.subsample_step]), dpi=100,
        vmin=-5, vmax=2, cmap=plt.get_cmap('Purples'), origin='lower')
    export_exact_image(args.name+'-delta-mask.png', pixel_mask[redshift_order][::args.subsample_step], dpi=100,
        origin='lower')

    print 'Computing mean delta...'

    mask_params = (params_a > .1) & (params_a < 10) & (params_b > -10) & (params_b < 10)

    delta_mean = ma.average(delta_flux[mask_params], axis=0)
    delta_mean_weighted = ma.average(delta_flux[mask_params], weights=delta_weight[mask_params], axis=0)
    delta_mean_ivar_weighted = ma.average(delta_flux[mask_params], weights=delta_ivar[mask_params], axis=0)

    plt.figure(figsize=(12,9))
    plt.plot(wave, delta_mean, label='Unweighted Mean')
    plt.plot(wave, delta_mean_weighted, label='LSS weighted Mean')
    plt.plot(wave, delta_mean_ivar_weighted, label='Ivar weighted Mean')
    # plt.ylim(0.06*np.array([-1,1]))
    plt.xlabel(r'Observed Wavelength ($\AA$)')
    plt.ylabel(r'Delta Mean')
    plt.grid()
    plt.legend()
    plt.savefig(args.name+'-lssweighted-delta-mean.png', dpi=100, bbox_inches='tight')
    plt.close()

    if args.dont_save:
        return -1

    outfile = h5py.File(args.name+'-delta.hdf5', 'w')
    # copy attributes from input files
    for attr_key in skim.attrs:
        outfile.attrs[attr_key] = skim.attrs[attr_key]
    # it's okay to overwrite the few that were already copied, I added a few attr to the combined
    # skim file and dont want to run the whole chain just yet
    for attr_key in linear_continuum.attrs:
        outfile.attrs[attr_key] = linear_continuum.attrs[attr_key]
    # create los group
    lines_of_sight = outfile.create_group('lines_of_sight')

    outfile.create_dataset('delta_mean', data=delta_mean.data)

    # loop over targets
    progress_bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(quasar_redshifts)).start()
    for i, z in enumerate(quasar_redshifts):
        progress_bar.update(i)

        if not mask_params[i]:
            # print 'fit param outside nominal range'
            continue

        z = quasar_redshifts[i]
        a = params_a[i]
        b = params_b[i]
        norm_i = norm[i]
        meta = skim['meta'][i]

        assert norm_i > 0

        ra = float(meta['ra'])
        dec = float(meta['dec'])
        thing_id = meta['thing_id']
        plate = meta['plate']
        mjd = meta['mjd']
        fiber = meta['fiber']

        # save to hdf5 file
        los = lines_of_sight.create_group(str(thing_id))
        los.attrs['plate'] = plate
        los.attrs['mjd'] = mjd
        los.attrs['fiber'] = fiber
        los.attrs['ra'] = ra
        los.attrs['dec'] = dec
        los.attrs['z'] = z
        los.attrs['p0'] = a
        los.attrs['p1'] = b

        los.create_dataset('loglam', data=loglam, dtype='f4')
        los.create_dataset('delta', data=(delta_flux[i]-delta_mean_weighted), dtype='f8')
        los.create_dataset('weight', data=delta_weight[i], dtype='f8')
        los.create_dataset('r_comov', data=np.zeros_like(loglam), dtype='f4')
        # los.create_dataset('ivar', data=ivar[i]/(norm_i*norm_i), dtype='f4')
        los.create_dataset('ivar', data=delta_ivar[i], dtype='f4')

    outfile.close()
    progress_bar.finish()


if __name__ == '__main__':
    main()
