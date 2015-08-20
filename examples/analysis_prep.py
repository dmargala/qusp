#!/usr/bin/env python
import argparse

import numpy as np
import numpy.ma as ma
import h5py

import qusp

import matplotlib.pyplot as plt

# def sum_chunk(x, chunk_size, axis=-1):
#     shape = x.shape
#     if axis < 0:
#         axis += x.ndim
#     shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
#     x = x.reshape(shape)
#     return x.sum(axis=axis+1)
def combine_pixels(loglam, flux, ivar, num_combine, trim_front=True):
    """
    Combines neighboring pixels of inner most axis using an ivar weighted average
    """
    shape = flux.shape
    num_pixels = flux.shape[-1]
    assert len(loglam) == num_pixels

    ndim = flux.ndim
    new_shape = shape[:ndim-1] + (-1, num_combine)

    num_leftover = num_pixels % num_combine
    s = slice(num_leftover,None) if trim_front else slice(0,-num_leftover)

    flux = flux[...,s].reshape(new_shape)
    ivar = ivar[...,s].reshape(new_shape)
    loglam = loglam[s].reshape(-1, num_combine)

    flux, ivar = ma.average(flux, weights=ivar, axis=ndim, returned=True)
    loglam = ma.average(loglam, axis=1)

    return loglam, flux, ivar

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## targets to fit
    parser.add_argument("-i", "--input", type=str, default=None,
        help="target list")
    parser.add_argument("-o", "--output", type=str, default=None,
        help="output file name")
    parser.add_argument("--num-combine", type=int, default=3,
        help="number of pixels to combine")
    parser.add_argument("--wave-min", type=float, default=3600,
        help="minimum observed wavelength")
    args = parser.parse_args()

    # import data
    skim = h5py.File(args.input, 'r')

    skim_norm = skim['norm'][:][:,np.newaxis]
    assert not np.any(skim_norm <= 0)

    skim_flux = np.ma.MaskedArray(skim['flux'][:], mask=skim['mask'][:])/skim_norm
    skim_ivar = np.ma.MaskedArray(skim['ivar'][:], mask=skim['mask'][:])*skim_norm*skim_norm
    skim_loglam = skim['loglam'][:]
    skim_wave = np.power(10.0, skim_loglam)

    good_waves = skim_wave > args.wave_min

    print 'Combining input pixels...'
    loglam, flux, ivar = combine_pixels(skim_loglam[good_waves], skim_flux[:,good_waves], skim_ivar[:,good_waves], args.num_combine)
    wave = np.power(10.0, loglam)

    outfile = h5py.File(args.output+'.hdf5', 'w')
    # save pixel flux, ivar, and mask
    outfile.create_dataset('flux', data=flux.data, compression="gzip")
    outfile.create_dataset('ivar', data=ivar.data, compression="gzip")
    outfile.create_dataset('mask', data=ivar.mask, compression="gzip")
    # save uniform wavelength grid
    outfile.create_dataset('loglam', data=loglam, compression="gzip")
    # save redshifts from input target list
    outfile.copy(skim['z'], 'z')
    # save additional quantities
    outfile.copy(skim['norm'], 'norm')
    # save meta data
    outfile.copy(skim['meta'], 'meta')
    # copy attrs
    for attr_key in skim.attrs:
        outfile.attrs[attr_key] = skim.attrs[attr_key]
    outfile.attrs['coeff0'] = loglam[0]
    outfile.attrs['coeff1'] = args.num_combine*1e-4
    outfile.attrs['max_fid_index'] = len(loglam)
    outfile.attrs['wave_min'] = args.wave_min

    outfile.close()

    # verify combined pixels
    print 'Computing mean and variance of input pixels...'
    skim_flux_mean = np.ma.average(skim_flux, axis=0, weights=skim_ivar)
    skim_flux_var = np.ma.average((skim_flux-skim_flux_mean)**2, axis=0, weights=skim_ivar)

    print 'Computing mean and variance of combined pixels...'
    flux_mean = np.ma.average(flux, axis=0, weights=ivar)
    flux_var = np.ma.average((flux-flux_mean)**2, axis=0, weights=ivar)

    print 'Making comparison plots...'
    plt.figure(figsize=(12,9))
    plt.plot(skim_wave, skim_flux_mean, label='Pipeline pixels')
    plt.plot(wave, flux_mean, label='Analysis pixels')
    plt.ylim(0.5, 1.5)
    plt.ylabel(r'Normalized Flux Mean (arb. units)')
    plt.xlabel(r'Observed Wavelength ($\AA$)')
    plt.legend()
    plt.grid()
    plt.savefig(args.output+'-flux-mean.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.plot(skim_wave, skim_flux_var, label='Pipeline pixels')
    plt.plot(wave, flux_var, label='Analysis pixels')
    plt.ylim(0, 0.45)
    plt.ylabel('Normalized Flux Variance (arb. units)')
    plt.xlabel(r'Observed Wavelength ($\AA$)')
    plt.legend()
    plt.grid()
    plt.savefig(args.output+'-flux-var.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.plot(skim_wave, np.sum(skim_ivar, axis=0), label='Pipeline pixels')
    plt.plot(wave, np.sum(ivar, axis=0), label='Analysis pixels')
    plt.ylabel('Inv. Var. Total (arb. units)')
    plt.xlabel(r'Observed Wavelength ($\AA$)')
    plt.legend()
    plt.grid()
    plt.savefig(args.output+'-ivar-total.png', dpi=100, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
