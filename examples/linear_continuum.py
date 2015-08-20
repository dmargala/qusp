#!/usr/bin/env python
import argparse

import numpy as np
import numpy.ma as ma
import h5py

import qusp

import matplotlib.pyplot as plt
import scipy.interpolate

from sklearn import linear_model


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## targets to fit
    parser.add_argument("--name", type=str, default=None,
        help="skim file name")
    parser.add_argument("--abs-beta", type=float, default=3.92,
        help='absorption redshift scaling power')
    parser.add_argument("--abs-alpha", type=float, default=0.0018,
        help='absorption coefficient')
    parser.add_argument("--forest-wave-ref", type=float, default=1185.0,
        help='forest wave reference')
    args = parser.parse_args()

    # import data
    forest_skim = h5py.File(args.name+'-forest.hdf5', 'r')
    forest_flux = np.ma.MaskedArray(forest_skim['flux'][:], mask=forest_skim['mask'][:])
    forest_ivar = np.ma.MaskedArray(forest_skim['ivar'][:], mask=forest_skim['mask'][:])
    forest_loglam = forest_skim['loglam'][:]
    forest_wave = np.power(10.0, forest_loglam)

    forest_norm = forest_skim['norm'][:]

    quasar_redshifts = forest_skim['z'][:]
    redshift_order = np.argsort(quasar_redshifts)

    wave_lya = forest_skim.attrs['wave_lya']

    forest_pixel_redshifts = (1.0 + quasar_redshifts[:,np.newaxis])*forest_wave/wave_lya - 1.0

    print 'Input data shape: ', forest_pixel_redshifts.shape

    #### Method 1, find which mean flux slice to use for which pixel
    ## uses: shifted_rows, shifted_cols, flux.shape, forest_flux, forest_weight, args.subsample_step
    ##  redshift_order, forest_pixel_redshifts
    print 'Starting linear continuum fit ...'
    num_forests, num_forest_waves = forest_flux.shape


    print 'Building model matrix...'

    log_forest_wave_ratio = np.log(forest_wave/args.forest_wave_ref)

    # raveled_weights = np.ones_like(forest_ivar).ravel()#np.sqrt(forest_ivar/(1.0+forest_ivar*0.055)).ravel()
    num_params = 2
    param_coefs = np.tile(np.vstack((np.ones(num_forest_waves), log_forest_wave_ratio)).reshape((-1,), order='F'), num_forests)
    param_rows = np.repeat(np.arange(num_forests*num_forest_waves), num_params)
    param_cols = np.vstack((np.repeat(np.arange(num_forests)*num_params, num_forest_waves),
                      np.repeat(np.arange(num_forests)*num_params + 1, num_forest_waves))).reshape((-1,), order='F')

    # num_params = 1
    # param_coefs = np.tile(np.ones(num_forest_waves), num_forests)
    # param_rows = np.arange(num_forests*num_forest_waves)
    # param_cols = np.repeat(np.arange(num_forests), num_forest_waves)

    print 'Param coef shapes: ', param_coefs.shape, param_rows.shape, param_cols.shape

    #### Add continuum coefficients
    cont_coefs = np.tile(np.ones(num_forest_waves), num_forests)
    cont_rows = np.arange(num_forests*num_forest_waves)
    cont_cols = np.tile(np.arange(num_forest_waves), num_forests)

    print 'Continuum coef shapes: ', cont_coefs.shape, cont_rows.shape, cont_cols.shape

    #### Add absorption coefficients
    abs_coefs = args.abs_alpha*np.power(1+forest_pixel_redshifts, args.abs_beta)

    # forest_min_z = 1.9
    # forest_max_z = 3.5
    # forest_dz = 0.1
    # num_z_bins = int((forest_max_z-forest_min_z)/forest_dz)
    # fz_zbin_indices = np.floor((forest_pixel_redshifts.ravel() - forest_min_z)/forest_dz).astype(int)
    #
    # print fz_zbin_indices.shape
    # print fz_zbin_indices

    # lo_coef = forest_pixel_redshifts - fz_zbin_indices*dz
    # hi_coef = forest_dz-lo_coef
    # abs_coefs = np.vstack((lo_coef,hi_coef)).reshape((-1,),order='F')
    # abs_cols = fz_zbin_indices
    # abs_rows = np.repeat(np.arange(num_forests*num_forest_waves), 2)

    # abs_coefs = np.ones(num_forest_waves*num_forests)
    # abs_rows = np.arange(num_forests*num_forest_waves)
    # abs_cols = fz_zbin_indices

    # print abs_coefs.shape

    model_coefs = np.concatenate((cont_coefs, param_coefs))
    model_rows = np.concatenate((cont_rows, param_rows))
    model_cols = np.concatenate((cont_cols, num_forest_waves+param_cols))

    print 'Model coef shapes: ', model_coefs.shape, model_rows.shape, model_cols.shape

    model_matrix = scipy.sparse.csc_matrix((model_coefs, (model_rows, model_cols)), shape=(num_forests*num_forest_waves,num_forest_waves+num_params*num_forests))

    print 'Model matrix shape: ', model_matrix.shape

    model_y = ma.log(ma.masked_where(forest_flux <= 0, forest_flux)) + abs_coefs

    print 'y shape, num masked pixels: ', model_y.shape, np.sum(model_y.mask)

    # valid = ~model_y.mask.ravel()

    regr = linear_model.LinearRegression(fit_intercept=False)
    print ('... performing fit using %s ...\n' % regr)

    # regr.fit(model_matrix[valid], model_y.ravel()[valid])
    regr.fit(model_matrix, model_y.ravel())
    soln = regr.coef_
    continuum = np.exp(soln[:num_forest_waves])
    # absorption = soln[num_forest_waves:2*num_forest_waves]
    params_a = np.exp(soln[num_forest_waves:num_forest_waves+num_params*num_forests:num_params])
    params_b = soln[num_forest_waves+1:num_forest_waves+num_params*num_forests:num_params]
    # mean_transmission = np.exp(soln[num_forest_waves+num_params*num_forests:])

    print 'Number of continuum params: ', continuum.shape

    outfile = h5py.File(args.name+'-linear-continuum.hdf5', 'w')
    # copy attributes from input file
    for attr_key in forest_skim.attrs:
        outfile.attrs[attr_key] = forest_skim.attrs[attr_key]
    # save args
    outfile.attrs['abs_alpha'] = args.abs_alpha
    outfile.attrs['abs_beta'] = args.abs_beta
    outfile.attrs['forest_wave_ref'] = args.forest_wave_ref
    # save fit results
    outfile.create_dataset('params_a', data=params_a, compression="gzip")
    outfile.create_dataset('params_b', data=params_b, compression="gzip")
    outfile.create_dataset('continuum', data=continuum, compression="gzip")
    outfile.create_dataset('continuum_wave', data=forest_wave, compression="gzip")
    outfile.close()

    # plt.figure(figsize=(12,9))
    # plt.plot(np.linspace(forest_min_z, forest_max_z, num_z_bins), mean_transmission, c='k')
    # plt.ylabel(r'z')
    # plt.xlabel(r'Mean F(z)')
    # plt.grid()
    # plt.savefig(args.name+'-linear-mean-transmission.png', dpi=100, bbox_inches='tight')
    # plt.close()

    plt.figure(figsize=(12,9))
    plt.step(forest_wave, continuum, c='k', where='mid')

    def draw_example(i, **kwargs):
        print quasar_redshifts[i]
        plt.scatter(forest_wave, forest_norm[i]*forest_flux[i], marker='+', **kwargs)
        plt.plot(forest_wave, forest_norm[i]*params_a[i]*np.exp(params_b[i]*log_forest_wave_ratio)*continuum, **kwargs)
    # draw_example(1, color='blue')
    # draw_example(10, color='green')
    # draw_example(100, color='red')
    plt.xlim(forest_wave[0], forest_wave[-1])
    plt.ylabel(r'Continuum (arb. units)')
    plt.xlabel(r'Rest Wavelength ($\AA$)')
    plt.grid()
    plt.savefig(args.name+'-linear-continuum.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.hist(params_a, bins=np.linspace(-0, 3, 51), histtype='stepfilled', alpha=0.5)
    plt.xlabel('a')
    plt.grid()
    plt.savefig(args.name+'-linear-param-a-dist.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.hist(params_b, bins=np.linspace(-20, 20, 51), histtype='stepfilled', alpha=0.5)
    plt.xlabel('b')
    plt.grid()
    plt.savefig(args.name+'-linear-param-b-dist.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.scatter(params_a, params_b, marker='+')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.ylim(-20,20)
    plt.xlim(0,3)
    plt.grid()
    plt.savefig(args.name+'-linear-param-scatter.png', dpi=100, bbox_inches='tight')
    plt.close()

    # rest and obs refer to pixel grid
    print 'Estimating deltas in forest frame...'
    model_flux = params_a[:,np.newaxis]*np.power(forest_wave/args.forest_wave_ref, params_b[:,np.newaxis])*continuum*np.exp(-abs_coefs)
    delta_flux_rest = forest_flux/model_flux - 1.0
    delta_ivar_rest = forest_ivar*(model_flux*model_flux)

    print 'Shifting deltas to observed frame...'

    shifted_rows = forest_skim['shifted_rows'][:]
    shifted_cols = forest_skim['shifted_cols'][:]
    shifted_loglam = forest_skim['shifted_loglam'][:]

    delta_flux_obs = ma.empty((num_forests, len(shifted_loglam)))
    delta_ivar_obs = ma.empty_like(delta_flux_obs)

    delta_flux_obs[shifted_rows, shifted_cols] = delta_flux_rest
    delta_ivar_obs[shifted_rows, shifted_cols] = delta_ivar_rest

    print 'Plotting mean delta...'

    mask_params = (params_a > .01) & (params_a < 100) & (params_b > -20) & (params_b < 20)

    print 'Number with okay params: %d' % np.sum(mask_params)
    delta_flux_mean = ma.average(delta_flux_obs[mask_params], axis=0, weights=delta_ivar_obs[mask_params])

    plt.figure(figsize=(12,9))
    plt.plot(np.power(10.0, shifted_loglam), delta_flux_mean)
    # plt.ylim(0.06*np.array([-1,1]))
    plt.xlabel(r'Observed Wavelength ($\AA$)')
    plt.ylabel(r'Delta Mean')
    plt.grid()
    plt.savefig(args.name+'-linear-delta-mean.png', dpi=100, bbox_inches='tight')
    plt.close()

    delta_flux_var = ma.average((delta_flux_obs[mask_params] - delta_flux_mean)**2, axis=0, weights=delta_ivar_obs[mask_params])
    plt.figure(figsize=(12,9))
    plt.plot(np.power(10.0, shifted_loglam), delta_flux_var)
    plt.ylim(0,0.5)
    plt.xlabel(r'Observed Wavelength ($\AA$)')
    plt.ylabel(r'Delta Variance')
    plt.grid()
    plt.savefig(args.name+'-linear-delta-var.png', dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
