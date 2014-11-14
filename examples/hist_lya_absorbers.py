#!/usr/bin/env python
"""
"""

import argparse
import qusp

import numpy as np

import matplotlib.pyplot as plt

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
    parser.add_argument("--output", type=str, default="absorber_redshifts.png",
        help="output file name")
    qusp.Paths.add_args(parser)
    qusp.target.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    # read target list
    targets = qusp.target.load_targets_from_args(args, fields = [('z', float, args.z_col)])

    forest_min = qusp.wavelength.Wavelength(args.forest_min)
    forest_max = qusp.wavelength.Wavelength(args.forest_max)
    wave_lya = qusp.wavelength.Wavelength(args.wave_lya)

    absorber_redshifts = []

    # loop over targets
    for target, combined in qusp.target.get_combined_spectra(targets, boss_path=paths.boss_path):

        # print target.to_string(), target['z']

        obs_min = forest_min.observed(target['z'])
        obs_max = forest_max.observed(target['z'])

        pixel_min = combined.find_pixel(obs_min, clip=True)
        if pixel_min == -1:
            pixel_min = 0
        pixel_max = combined.find_pixel(obs_max, clip=True)
        if pixel_max == combined.npixels:
            pixel_max = combined.npixels-1

        absorber_z = combined.wavelength[pixel_min:pixel_max+1]/wave_lya - 1

        absorber_redshifts += absorber_z.tolist()

    fig = plt.figure(figsize=(8,6))
    plt.hist(absorber_redshifts, bins=50, linewidth=.1, alpha=.5)
    plt.xlabel(r'Absorber Redshifts')
    plt.grid()
    fig.savefig(args.output, bbox_inches='tight')

if __name__ == '__main__':
    main()
