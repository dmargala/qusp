#!/usr/bin/env python
"""
"""

import argparse
import qusp

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="more verbose output")
    parser.add_argument("--forest-min", type=float, default=1040,
        help="wavelength of lya forest min")
    parser.add_argument("--forest-max", type=float, default=1200,
        help="wavelength of lya forest max")
    parser.add_argument("--wave-lya", type=float, default=1216,
        help="wavelength of lya line")
    parser.add_argument("--z-col", type=int, default=1,
        help="redshift column index")
    parser.add_argument("--output", type=str, default="",
        help="output file name")
    parser.add_argument("--unweighted", action="store_true",
        help="don't use ivar weighting")
    qusp.Paths.add_args(parser)
    qusp.target.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    # read target list
    target_list = qusp.target.load_target_list_from_args(args, 
        fields=[('z', float, args.z_col)])

    forest_min = qusp.wavelength.Wavelength(args.forest_min)
    forest_max = qusp.wavelength.Wavelength(args.forest_max)
    wave_lya = qusp.wavelength.Wavelength(args.wave_lya)

    absorber_redshifts = []
    absorber_weights = []
    absorber_transmissions = []

    norm_min = qusp.wavelength.Wavelength(1275)
    norm_max = qusp.wavelength.Wavelength(1285)

    # loop over targets
    for target, combined in qusp.target.get_combined_spectra(target_list, boss_path=paths.boss_path):
        continuum = combined.mean_flux(norm_min.observed(target['z']), norm_max.observed(target['z']))
        # determine observed frame forest window
        obs_min = forest_min.observed(target['z'])
        obs_max = forest_max.observed(target['z'])
        # find pixels values corresponding to this window
        pixel_min = combined.find_pixel(obs_min, clip=True)
        pixel_max = combined.find_pixel(obs_max, clip=True)
        forest_slice = slice(pixel_min, pixel_max+1)
        # calculate absorber redshifts and weights
        absorber_z = combined.wavelength[forest_slice]/wave_lya - 1
        absorber_weight = combined.ivar.values[forest_slice]
        absorber_transmission = combined.flux.values[forest_slice]/continuum
        # save this absorbers for this target
        absorber_redshifts.append(absorber_z)
        absorber_weights.append(absorber_weight)
        absorber_transmissions.append(absorber_transmission)

    absorber_redshifts = np.concatenate(absorber_redshifts)
    if args.unweighted:
        absorber_weights = np.ones_like(absorber_redshifts)
    else:
        absorber_weights = np.concatenate(absorber_weights)
    absorber_transmissions = np.concatenate(absorber_transmissions)


    if args.verbose:
        print 'Number of absorbers: %d' % absorber_redshifts.shape[0]
        print 'Mean number per target: %.2f' % (absorber_redshifts.shape[0]/len(target_list))
        print 'Mean absorber redshift: %.4f' % np.mean(absorber_redshifts)
        print 'Mean transimission: %.4f' % np.mean(absorber_transmissions)

    fig = plt.figure(figsize=(8,6))

    # plt.plot(absorber_redshifts, absorber_transmissions, 'o', mec='none', alpha=.05)
    # plt.grid()

    zbins = np.arange(absorber_redshifts.min(),absorber_redshifts.max()+.01,.01)
    # digitized = np.digitize(absorber_redshifts, zbins)
    # bin_means = [daabsorber_transmissions[digitized == i].mean() for i in range(1, len(zbins))]

    bin_means = scipy.stats.binned_statistic(absorber_redshifts, 
        absorber_transmissions, statistic='mean', bins=zbins)[0]
    bin_centers = (zbins[:-1]+zbins[1:])/2
    print bin_centers.shape
    plt.hist2d(absorber_redshifts, absorber_transmissions, bins=[zbins,np.linspace(-2,4,100+1)], cmap='Greens')
    plt.plot(bin_centers, bin_means, '-')
    plt.colorbar()

    if args.output:
        fig.savefig(args.output, bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    main()
