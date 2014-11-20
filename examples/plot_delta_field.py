#!/usr/bin/env python

import argparse

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import h5py
import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-i","--input", type=str, default=None,
        help="fitspec results file")
    parser.add_argument("-o","--output", type=str, default=None,
        help="output filename base")
    args = parser.parse_args()

    infile = h5py.File(args.input)

    ra_list = []
    dec_list = []
    redshift_list = []

    for (target_string, h5group) in infile['delta_field'].iteritems():
        ra = np.radians(h5group.attrs['ra'])
        dec = np.radians(h5group.attrs['dec'])
        z = h5group.attrs['z'] 
        #redshift = h5group['absorber_z'].value
        #redshift = np.array([max(1.9, 1026*(1+3.5)/1216-1), z])

        redshift = np.array([z])

        redshift_list.append(redshift)
        ra_list.append(ra*np.ones_like(redshift))
        dec_list.append(dec*np.ones_like(redshift))

        print target_string, ra, dec, redshift

    # flatten lists
    ra = np.concatenate(ra_list)
    dec = np.concatenate(dec_list)
    redshift = np.concatenate(redshift_list)

    x = redshift * np.cos(ra) * np.sin(dec)
    y = redshift * np.sin(ra) * np.sin(dec)
    z = redshift * np.cos(dec)

    zmax = 7
    max_scale = 1.2

    if args.output:

        fig = plt.figure(figsize=(14,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        ax.scatter(x, y, z, marker='.', s=.1)

        ax.scatter(x, y, -max_scale*zmax, zdir='z', marker='.', s=.1)
        ax.scatter(y, z, -max_scale*zmax, zdir='x', marker='.', s=.1)
        ax.scatter(x, z, +max_scale*zmax, zdir='y', marker='.', s=.1)

        def plot_shell(r):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)

            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='r', alpha=.05, edgecolor='None')

            x = r * np.cos(u)
            y = r* np.sin(u)
            ax.plot(x, y, zs=-max_scale*zmax, zdir='z', color='r', alpha=.2)
            ax.plot(x, y, zs=-max_scale*zmax, zdir='x', color='r', alpha=.2)
            ax.plot(x, y, zs=+max_scale*zmax, zdir='y', color='r', alpha=.2)

        for r in np.arange(1, zmax+1, 1):
            plot_shell(r)

        ax.set_xlim3d(-max_scale*zmax, max_scale*zmax)
        ax.set_ylim3d(-max_scale*zmax, max_scale*zmax)
        ax.set_zlim3d(-max_scale*zmax, max_scale*zmax)

        #ax.view_init(0, 0)

        fig.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()

