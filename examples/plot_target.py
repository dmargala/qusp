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
    parser.add_argument("--wave-min", type=float, default=3600,
        help="wavelength min")
    parser.add_argument("--wave-max", type=float, default=10500,
        help="wavelength max")
    parser.add_argument("--outdir", type=str, default=None,
        help="output filename")
    parser.add_argument("--target-index", type=int, default=None)
    qusp.paths.Paths.add_args(parser)
    qusp.target.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))

    # wavelength range limits
    wave_min = qusp.wavelength.Wavelength(args.wave_min)
    wave_max = qusp.wavelength.Wavelength(args.wave_max)

    # read target
    if args.targets and args.target_index is not None:
        target_list = qusp.target.load_target_list_from_args(args)
        target = target_list[args.target_index]
    elif args.target:
        target = qusp.Target.from_string(args.target)
    elif args.plate and args.mjd and args.fiber:
        target = qusp.Target.from_plate_mjd_fiber(args.plate, args.mjd, args.fiber)
    else:
        raise RuntimeError('Invalid target specification.')

    # load target's spectrum
    combined = qusp.target.get_combined_spectrum(target, paths)

    # determine observed spectrum window
    forest = combined.trim_range(wave_min, wave_max)

    ymin = 0
    ymax = 42

    fig = plt.figure(figsize=(14, 6))

    badpixels = np.where(combined.ivar.values == 0)

    plt.plot(combined.wavelength, combined.flux.values, color='blue', lw=.5)#, marker='+', markersize=3, lw=0)

    y_err = 1/np.sqrt(combined.ivar.values)
    #y_err_lower = combined.flux.values - y_err
    #y_err_upper = combined.flux.values + y_err
    #plt.fill_between(combined.wavelength, y_err_lower, y_err_upper, facecolor='gray', alpha=.5, lw=0)
    #plt.errorbar(combined.wavelength, combined.flux.values, y_err, color='blue', marker='+', ls='None', lw=.2, mew=0)

    plt.xlim([wave_min, wave_max])
    plt.ylim([ymin, ymax])

    # quasar_lines = qusp.wavelength.load_wavelengths('quasar')
    # redshifted_quasar_lines = []
    # for line in quasar_lines:
    #     redshifted_quasar_lines.append(qusp.wavelength.LabeledWavelength(line*(1+target['z']), line.label))
    # qusp.wavelength.draw_lines(redshifted_quasar_lines, 0.895, -0.05, ls='--', color='black')
    # qusp.wavelength.draw_lines(qusp.wavelength.load_wavelengths('sky', ignore_labels=True), 
    #     0.01, 0.1, color='magenta', alpha=.3)

    plt.title(target.to_string())
    plt.ylabel(r'Flux $(10^{-17} erg/cm^2/s/\AA)$')
    plt.xlabel(r'Observed Wavlength $(\AA)$')

    plt.grid()

    filename = target.to_string()+'.png'
    if args.outdir:
        filename = os.path.join(args.output, filename)
    fig.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':
    main()
