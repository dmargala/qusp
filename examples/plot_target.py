#!/usr/bin/env python

import argparse
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py

import qusp


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="more verbose output")
    parser.add_argument("--plate", type=int, default=None,
        help="plate number")
    parser.add_argument("--mjd", type=int, default=None,
        help="mjd")
    parser.add_argument("--fiber", type=int, default=None,
        help="fiber id")
    parser.add_argument("--target", type=str, default=None,
        help="target string")
    parser.add_argument("-z", type=float, default=0,
        help="redshift")
    parser.add_argument("--z-col", type=int, default=1,
        help="redshift column in target list file")
    parser.add_argument("--forest-min", type=float, default=1040,
        help="wavelength of lya forest min")
    parser.add_argument("--forest-max", type=float, default=1200,
        help="wavelength of lya forest max")
    parser.add_argument("--continuum", type=str, default="",
        help="continuum to use")
    parser.add_argument("--output", type=str, default=None,
        help="output filename")
    parser.add_argument("--target-index", type=int, default=None)
    qusp.paths.Paths.add_args(parser)
    qusp.target.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))

    # define forest range and lya wavelength
    forest_min = qusp.wavelength.Wavelength(args.forest_min)
    forest_max = qusp.wavelength.Wavelength(args.forest_max)
    wave_lya = qusp.wavelength.LabeledWavelength(1216, 'Lya')
    wave_lyb = qusp.wavelength.LabeledWavelength(1026, 'Lyb')
    wave_lyg = qusp.wavelength.LabeledWavelength(972, 'Lyg')
    wave_norm = qusp.wavelength.Wavelength(1280)

    # initialize continuum model
    if args.continuum:
        continuum_model = qusp.LinearFitContinuum(args.continuum)
    else:
        continuum_model = qusp.MeanFluxContinuum(forest_min, forest_max)

    # read target
    if args.targets and args.target_index is not None:
        target_list = qusp.target.load_target_list_from_args(args,
            fields=[('z', float, args.z_col)])
        target = target_list[args.target_index]
    elif args.target:
        target = qusp.Target.from_string(args.target)
        target['z'] = args.z
    elif args.plate and args.mjd and args.fiber:
        target = qusp.Target.from_plate_mjd_fiber(args.plate, args.mjd, args.fiber)
        target['z'] = args.z
    else:
        raise RuntimeError('Invalid target specification.')

    # load target's spectrum
    combined = qusp.target.get_combined_spectrum(target, paths)

    # determine observed frame forest window
    obs_forest_min = forest_min.observed(target['z'])
    obs_forest_max = forest_max.observed(target['z'])
    forest = combined.trim_range(obs_forest_min, obs_forest_max)

    continuum = continuum_model.get_continuum(target, combined)

    ymin = -5
    ymax = 20

    fig = plt.figure(figsize=(14, 6))

    plt.plot(continuum.wavelength, continuum.values, color='black', ls='-', lw=1.5)
    #plt.fill_between(continuum.wavelength, ymin, continuum.values, facecolor='gray', alpha=1, lw=0)

    #plt.fill_between(combined.wavelength, ymin, combined.flux.values, facecolor='gray', alpha=1, lw=0)
    plt.plot(combined.wavelength, combined.flux.values, color='blue', marker='+', markersize=3, lw=0)

    badpixels = np.where(combined.ivar.values == 0)

    y_err = 1/np.sqrt(combined.ivar.values)
    y_err_lower = combined.flux.values - y_err
    y_err_upper = combined.flux.values + y_err
    plt.fill_between(combined.wavelength, y_err_lower, y_err_upper, facecolor='gray', alpha=.5, lw=0)

    #plt.fill_between(forest.wavelength, ymin, forest.flux.values, facecolor='gray', alpha=1, lw=0)
    plt.plot(forest.wavelength, forest.flux.values, color='red', marker='+', markersize=3, lw=0)

    plt.xlim([wave_lyg.observed(target['z']), wave_norm.observed(target['z'])])
    plt.ylim([ymin, ymax])

    quasar_lines = qusp.wavelength.load_wavelengths('quasar')
    redshifted_quasar_lines = []
    for line in quasar_lines:
        redshifted_quasar_lines.append(qusp.wavelength.LabeledWavelength(line*(1+target['z']), line.label))
    qusp.wavelength.draw_lines(redshifted_quasar_lines, 0.895,-0.05,  ls='--', c='black')

    plt.title(target.to_string()+' (z = %.3f)' % target['z'])
    plt.ylabel(r'Flux $(10^{-17} erg/cm^2/s/\AA)$')
    plt.xlabel(r'Observed Wavlength $(\AA)$')

    plt.grid()

    filename = target.to_string()+'.png'
    if args.output:
        filename = args.output+'-'+filename
    fig.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':
    main()
