#!/usr/bin/env python
"""
"""

import argparse
import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    qusp.target.add_args(parser)
    qusp.Paths.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))

    # read target list
    targets = qusp.target.load_target_list_from_args(args,
        fields=[(band, float, i+1) for i,band in enumerate('ugriz')])

    # loop over targets
    for target, combined in qusp.target.get_combined_spectra(targets, boss_path=paths.boss_path):
        ab_mags = combined.flux.get_ab_magnitudes()
        print ' '.join(['%.4f' % ab_mags[band] for band in 'gri'])

if __name__ == '__main__':
    main()
