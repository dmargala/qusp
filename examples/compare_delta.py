#!/usr/bin/env python
import argparse

import numpy as np
import numpy.ma as ma
import h5py

import qusp

import matplotlib.pyplot as plt
import scipy.interpolate
import fitsio

class DeltaLOS(object):
    def __init__(self, thing_id):
        path = '/data/lya/deltas/delta-%d.fits' % thing_id
        hdulist = fitsio.FITS(path, mode=fitsio.READONLY)
        self.pmf = hdulist[1].read_header()['pmf']
        self.loglam = hdulist[1]['loglam'][:]
        self.wave = np.power(10.0, self.loglam)
        self.delta = hdulist[1]['delta'][:]
        self.weight = hdulist[1]['weight'][:]
        self.cont = hdulist[1]['cont'][:]
        self.msha = hdulist[1]['msha'][:]
        self.mabs = hdulist[1]['mabs'][:]
        self.ivar = hdulist[1]['ivar'][:]

        self.cf = self.cont*self.msha*self.mabs
        self.flux = (1+self.delta)*self.cf

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    ## targets to fit
    parser.add_argument("--name", type=str, default=None,
        help="target list")
    parser.add_argument("--gamma", type=float, default=3.8,
        help="LSS growth and redshift evolution of mean absorption gamma")
    parser.add_argument("--index", type=int, default=1000,
        help="target index")
    parser.add_argument("--pmf", type=str, default=None,
        help="target plate-mjd-fiber string")
    args = parser.parse_args()

    print 'Loading forest data...'

    # import data
    skim = h5py.File(args.name+'.hdf5', 'r')

    if args.pmf:
        plate, mjd, fiber = [int(val) for val in args.pmf.split('-')]
        index = np.where((skim['meta']['plate'] == plate) & (skim['meta']['mjd'] == mjd) & (skim['meta']['fiber'] == fiber))[0][0]
    else:
        index = args.index

    flux = np.ma.MaskedArray(skim['flux'][index], mask=skim['mask'][index])
    ivar = np.ma.MaskedArray(skim['ivar'][index], mask=skim['mask'][index])
    loglam = skim['loglam'][:]
    wave = np.power(10.0, loglam)

    z = skim['z'][index]
    norm = skim['norm'][index]
    meta = skim['meta'][index]

    linear_continuum = h5py.File(args.name+'-linear-continuum.hdf5', 'r')
    a = linear_continuum['params_a'][index]
    b = linear_continuum['params_b'][index]
    continuum = linear_continuum['continuum']
    continuum_wave = linear_continuum['continuum_wave']
    continuum_interp = scipy.interpolate.UnivariateSpline(continuum_wave, continuum, ext=1, s=0)
    abs_alpha = linear_continuum.attrs['abs_alpha']
    abs_beta = linear_continuum.attrs['abs_beta']

    forest_wave_ref = (1+z)*linear_continuum.attrs['forest_wave_ref']
    wave_lya = linear_continuum.attrs['wave_lya']

    forest_pixel_redshifts = wave/wave_lya - 1
    abs_coefs = abs_alpha*np.power(1+forest_pixel_redshifts, abs_beta)

    print 'flux 1280 Ang: %.2f' % norm
    print 'fit param a: %.2f' % a
    print 'fit param b: %.2f' % b

    def model_flux(a, b):
        return a*np.power(wave/forest_wave_ref, b)*continuum_interp(wave/(1+z))*np.exp(-abs_coefs)

    def chisq(p):
        mflux = model_flux(p[0], p[1])
        res = flux - mflux
        return ma.sum(res*res*ivar)/ma.sum(ivar)

    from scipy.optimize import minimize

    result = minimize(chisq, (a, b))
    a,b = result.x

    print 'fit param a: %.2f' % a
    print 'fit param b: %.2f' % b

    # rest and obs refer to pixel grid
    print 'Estimating deltas in forest frame...'

    mflux = model_flux(a,b)
    delta_flux = flux/mflux - 1.0
    delta_ivar = ivar*mflux*mflux

    forest_min_z = linear_continuum.attrs['forest_min_z']
    forest_max_z = linear_continuum.attrs['forest_max_z']
    forest_dz = 0.1
    forest_z_bins = np.arange(forest_min_z, forest_max_z + forest_dz, forest_dz)

    print 'Adjusting weights for pipeline variance and LSS variance...'

    var_lss = scipy.interpolate.UnivariateSpline(forest_z_bins, 0.05 + 0.06*(forest_z_bins - 2.0)**2, s=0)
    var_pipe_scale = scipy.interpolate.UnivariateSpline(forest_z_bins, 0.7 + 0.2*(forest_z_bins - 2.0)**2, s=0)

    delta_weight = delta_ivar*var_pipe_scale(forest_pixel_redshifts)
    delta_weight = delta_weight/(1 + delta_weight*var_lss(forest_pixel_redshifts))

    thing_id = meta['thing_id']
    pmf = '%s-%s-%s' % (meta['plate'],meta['mjd'],meta['fiber'])

    los = DeltaLOS(thing_id)

    my_msha = norm*a*np.power(wave/forest_wave_ref, b)
    my_wave = wave
    my_flux = norm*flux
    my_cf = my_msha*continuum_interp(wave/(1+z))*np.exp(-abs_coefs)
    my_ivar = ivar/(norm*norm)
    my_delta = delta_flux
    my_weight = delta_weight

    # mean_ratio = np.average(my_msha*continuum)/ma.average(los.msha*los.cont)
    # print mean_ratio

    plt.figure(figsize=(12,4))
    plt.plot(my_wave, my_flux, color='gray')

    my_dflux = ma.power(my_ivar, -0.5)
    plt.fill_between(my_wave, my_flux - my_dflux, my_flux + my_dflux, color='gray', alpha=0.5)

    plt.plot(my_wave, my_msha*continuum_interp(wave/(1+z)), label='My continuum', color='blue')
    plt.plot(los.wave, los.cont, label='Busca continuum', color='red')
    plt.plot(my_wave, my_cf, label='My cf', color='green')
    plt.plot(los.wave, los.cf, label='Busca cf', color='orange')
    plt.legend()
    plt.title(r'%s (%s), $z$ = %.2f' % (pmf, thing_id, z))
    plt.xlabel(r'Observed Wavelength ($\AA$)')
    plt.ylabel(r'Observed Flux')
    plt.xlim(los.wave[[0,-1]])
    plt.savefig(args.name+'-example-flux.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,4))
    my_delta_sigma = ma.power(delta_weight, -0.5)
    # plt.fill_between(my_wave, my_delta - my_delta_sigma, my_delta + my_delta_sigma, color='blue', alpha=0.1, label='My Delta')
    plt.scatter(my_wave, my_delta, color='blue', marker='+', label='My Delta')
    plt.plot(my_wave, +my_delta_sigma, color='blue', ls=':')
    plt.plot(my_wave, -my_delta_sigma, color='blue', ls=':')

    los_delta_sigma = ma.power(los.weight, -0.5)
    # plt.fill_between(los.wave, los.delta - los_delta_sigma, los.delta + los_delta_sigma, color='red', alpha=01, label='Busca Delta')
    plt.scatter(los.wave, los.delta, color='red', marker='+', label='Busca Delta')

    plt.plot(los.wave, +los_delta_sigma, color='red', ls=':')
    plt.plot(los.wave, -los_delta_sigma, color='red', ls=':')

    my_lss_sigma = np.sqrt(var_lss(forest_pixel_redshifts))
    plt.plot(my_wave, +my_lss_sigma, color='black', ls='--')
    plt.plot(my_wave, -my_lss_sigma, color='black', ls='--')

    # my_sn_sigma = np.sqrt(np.power(1 + forest_pixel_redshifts, 0.5*abs_beta))/10
    # plt.plot(my_wave, +my_sn_sigma, color='orange', ls='--')
    # plt.plot(my_wave, -my_sn_sigma, color='orange', ls='--')
    # import matplotlib.patches as mpatches
    #
    # blue_patch = mpatches.Patch(color='blue', alpha=0.3, label='My Delta')
    # red_patch = mpatches.Patch(color='red', alpha=0.3, label='Busca Delta')
    # plt.legend(handles=[blue_patch,red_patch])

    plt.title(r'%s (%s), $z$ = %.2f' % (pmf, thing_id, z))
    plt.ylim(-2,2)
    plt.xlim(los.wave[[0,-1]])

    plt.xlabel(r'Observed Wavelength ($\AA$)')
    plt.ylabel(r'Delta')
    plt.legend()
    plt.savefig(args.name+'-example-delta.png', dpi=100, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    main()
