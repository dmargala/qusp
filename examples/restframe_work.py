#!/usr/bin/env python
import argparse

import numpy as np
import numpy.ma as ma
import h5py

import qusp

import matplotlib.pyplot as plt
import scipy.interpolate

def export_exact_image(filename, img, dpi=100, **kwargs):
    h, w = img.shape
    fig = plt.figure(frameon=False)
    fig.set_size_inches(float(w)/dpi,float(h)/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto', interpolation='none', **kwargs)
    fig.savefig(filename, dpi=dpi)
    plt.close()

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    ## targets to fit
    parser.add_argument("-i", "--input", type=str, default=None,
        help="target list")
    parser.add_argument("-o", "--output", type=str, default=None,
        help="output file name")
    parser.add_argument("--subsample-step", type=int, default=1000,
        help="step size used for subsampling observations")
    parser.add_argument("--wave-lya", type=float, default=1216.0,
        help="lyman alpha wavelength")
    parser.add_argument("--forest-min-z", type=float, default=2,
        help="minimum forest pixel redshift")
    parser.add_argument("--forest-max-z", type=float, default=3.5,
        help="maximum forest pixel redshift")
    args = parser.parse_args()

    # import data
    skim = h5py.File(args.input, 'r')
    norm = skim['norm'][:][:,np.newaxis]
    flux = np.ma.MaskedArray(skim['flux'][:], mask=skim['mask'][:])
    ivar = np.ma.MaskedArray(skim['ivar'][:], mask=skim['mask'][:])
    loglam = skim['loglam'][:]
    wave = np.power(10.0, loglam)

    print 'Plotting quasar redshift distribution...'
    quasar_redshifts = skim['z'][:]
    redshift_order = np.argsort(quasar_redshifts)

    plt.figure(figsize=(12,9))
    quasar_min_z = 2.1
    quasar_max_z = 3.5
    quasar_dz = 0.025
    plt.hist(quasar_redshifts, bins=np.arange(quasar_min_z, quasar_max_z + quasar_dz, quasar_dz),
        histtype='stepfilled', alpha=0.5)
    plt.xlim(quasar_min_z - 0.1, quasar_max_z + 0.1)
    plt.ylabel(r'Number of Quasars per $\Delta z = %.3f$' % quasar_dz)
    plt.xlabel(r'Redshift $z$')
    plt.grid()
    plt.savefig(args.output+'-quasar-redshift-dist.png', dpi=100, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12,9))
    plt.hist(ivar.ravel(), bins=np.linspace(4, 200, 50),
        histtype='stepfilled', alpha=0.5)
    plt.ylabel(r'Number of pixels per bin')
    plt.xlabel(r'Inv. Variance')
    plt.grid()
    plt.savefig(args.output+'-ivar-dist.png', dpi=100, bbox_inches='tight')
    plt.close()

    ############################

    print 'Aligning observed pixels to forest pixels...'
    forest_lo = skim.attrs['forest_lo']
    forest_hi = skim.attrs['forest_hi']
    log10lam0 = skim.attrs['coeff0']
    coeff1 = skim.attrs['coeff1']

    forest_lo_index = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(forest_lo), coeff1=coeff1, log10lam0=log10lam0)
    forest_hi_index = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(forest_hi), coeff1=coeff1, log10lam0=log10lam0)

    forest_wave = qusp.wavelength.get_fiducial_wavelength(np.arange(forest_lo_index, forest_hi_index), coeff1, log10lam0)

    lo_indices = qusp.wavelength.get_fiducial_pixel_index_offset(np.log10(forest_wave[0]*(1 + quasar_redshifts)), coeff1, log10lam0)

    print lo_indices
    offsets = np.zeros_like(lo_indices)
    offsets[lo_indices > 0] = lo_indices[lo_indices > 0]
    print offsets

    num_forest_waves = len(forest_wave)
    num_rows = flux.shape[0]

    shifted_cols = np.mod(np.tile(np.arange(num_forest_waves), num_rows).reshape(num_rows, num_forest_waves) + lo_indices.reshape(-1,1), (num_forest_waves+offsets).reshape(-1,1))
    shifted_rows = np.repeat(np.arange(num_rows), num_forest_waves).reshape(num_rows, num_forest_waves)

    forest_flux = flux[shifted_rows, shifted_cols]
    forest_ivar = ivar[shifted_rows, shifted_cols]

    forest_pixel_redshifts = (1.0 + quasar_redshifts[:,np.newaxis])*forest_wave/args.wave_lya - 1.0

    forest_flux = ma.masked_where(forest_pixel_redshifts < args.forest_min_z, forest_flux, copy=False)
    forest_ivar = ma.masked_where(forest_pixel_redshifts < args.forest_min_z, forest_ivar, copy=False)

    forest_pixel_redshifts = ma.MaskedArray(forest_pixel_redshifts, mask=forest_flux.mask)

    print 'Alignment complete! ', forest_flux.shape

    outfile = h5py.File(args.output+'-forest.hdf5', 'w')
    # copy attributes from input file
    for attr_key in skim.attrs:
        outfile.attrs[attr_key] = skim.attrs[attr_key]
    # save args
    outfile.attrs['wave_lya'] = args.wave_lya
    outfile.attrs['forest_min_z'] = args.forest_min_z
    outfile.attrs['forest_max_z'] = args.forest_max_z
    # save pixel flux, ivar, and mask
    outfile.create_dataset('flux', data=forest_flux.data, compression="gzip")
    outfile.create_dataset('ivar', data=forest_ivar.data, compression="gzip")
    outfile.create_dataset('mask', data=forest_ivar.mask, compression="gzip")
    # save uniform wavelength grid
    outfile.create_dataset('loglam', data=np.log10(forest_wave), compression="gzip")
    # save info to undo shift
    outfile.create_dataset('shifted_rows', data=shifted_rows, compression="gzip")
    outfile.create_dataset('shifted_cols', data=shifted_cols, compression="gzip")
    outfile.create_dataset('shifted_loglam', data=loglam, compression="gzip")
    # save quasar redshifts
    outfile.copy(skim['z'], 'z')
    outfile.copy(skim['norm'], 'norm')
    outfile.copy(skim['meta'], 'meta')
    outfile.close()

    plt.figure(figsize=(12,9))
    extent = (forest_wave[0],forest_wave[-1],1,num_rows)
    plt.imshow(forest_pixel_redshifts[redshift_order][::args.subsample_step], aspect='auto',
               vmin=args.forest_min_z, vmax=args.forest_max_z, cmap=plt.get_cmap('YlGn'),
               origin='lower', extent=extent)
    plt.xlabel(r'Rest Wavelength ($\AA$)')
    plt.ylabel(r'Observation Index (sorted by $z$)')
    plt.colorbar()
    plt.savefig(args.output+'-forest-redshift-subsample.png', dpi=100, bbox_inches='tight')
    plt.close()

    print 'Saving exact images...'

    export_exact_image(args.output+'-forest-redshift.png', forest_pixel_redshifts[redshift_order][::args.subsample_step], dpi=100,
        vmin=args.forest_min_z, vmax=args.forest_max_z, cmap=plt.get_cmap('YlGn'), origin='lower')

    export_exact_image(args.output+'-forest-flux.png', forest_flux[redshift_order][::args.subsample_step], dpi=100,
        vmin=0, vmax=3, cmap=plt.get_cmap('YlGn'), origin='lower')

    export_exact_image(args.output+'-flux.png', flux[redshift_order][::args.subsample_step], dpi=100,
        vmin=0, vmax=3, cmap=plt.get_cmap('YlGn'), origin='lower')


if __name__ == '__main__':
    main()
