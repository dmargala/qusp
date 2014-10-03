#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-o","--output", type=str, default=None,
        help="hdf5 output filename")
    ## targets to fit
    parser.add_argument("-i","--input", type=str, default=None,
        help="target list")
    parser.add_argument("-n","--ntargets", type=int, default=0,
        help="number of targets to use, 0 for all")
    parser.add_argument("--z-col", type=int, default=3,
        help="redshift column of input targetlist")
    qusp.Paths.addArgs(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.fromArgs(args))

    # read target list
    fields = [('z',float,args.z_col)]
    targets = qusp.target.loadTargetData(args.input, fields=fields)
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)

    targets = sorted(targets[:ntargets], key=lambda target: (target['plate'],target['mjd'],target['fiber']))

    print ntargets

    currentObs = '%s-%s' % (targets[0]['plate'],targets[0]['mjd'])
    samePlateTargets = []
    for target in targets:
        plateMJD = '%s-%s' % (target['plate'],target['mjd'])
        if currentObs != plateMJD:
            if len(samePlateTargets) > 100:
                qusp.target.saveTargetData('%s-%s.txt' % (args.output,currentObs), samePlateTargets, 'z')
            samePlateTargets = []
            currentObs = plateMJD
        samePlateTargets.append(target)
    qusp.target.saveTargetData('%s-%s.txt' % (args.output,currentObs), samePlateTargets, 'z')

    # fig = plt.figure(figsize=(8,8))
    # ax = plt.subplot(111,polar=True)

    # for target, spPlate in qusp.target.readTargetPlates(paths.boss_path,targets):
    #     i = target['fiber'] - 1
        # taibeg = spPlate[0].header['TAI-BEG']
        # taiend = spPlate[0].header['TAI-END']
        # nexp = spPlate[0].header['NEXP']/4
        # ra = spPlate[0].header['RA']
        # dec = spPlate[0].header['DEC']

        # plateDiameter = (3.0*u.deg).to(u.rad)

        # ra = (spPlate[5].data[i]['ra']*u.deg).to(u.rad)
        # dec = (spPlate[5].data[i]['dec']*u.deg).to(u.rad)

        # ax.scatter(ra, target['z'], alpha=.5, marker='.', lw=0)


        #bars = ax.bar(ra, target['z'], width=plateDiameter.value, bottom=0.5, alpha=.1)

    # plt.tick_params(labelleft='off')
    # plt.ylim([0,4])
    # plt.show()
    # Use custom colors and opacity
    # for r, bar in zip(radii, bars):
    #     bar.set_facecolor(plt.cm.jet(r / 10.))
    #     bar.set_alpha(0.5)

if __name__ == '__main__':
    main()