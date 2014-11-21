#!/usr/bin/env python

import argparse

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import h5py
import qusp

from astropy import units as u
from astropy.coordinates import SkyCoord

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
    parser.add_argument("--max-rows", type=int, default=0,
        help="max number of entries to plot")
    parser.add_argument("--elev", type=float, default=None,
        help="elevation angle")
    parser.add_argument("--azim", type=float, default=None,
        help="azimuthal angle")
    parser.add_argument("--stride", type=int, default=1,
        help="stride")
    args = parser.parse_args()

    infile = h5py.File(args.input)

    ra_list = []
    dec_list = []
    redshift_list = []

    counter = 0

    for (target_string, h5group) in infile['delta_field'].iteritems():
        counter += 1
        if (counter % 10000) == 0:
            print ' processing target # %d ...' % (counter)
        if args.max_rows and counter > args.max_rows:
            break
        ra = np.radians(h5group.attrs['ra'])
        dec = np.radians(h5group.attrs['dec'])
        z = h5group.attrs['z'] 
        #redshift = h5group['absorber_z'].value
        #redshift = np.array([max(1.9, 1026*(1+3.5)/1216-1), z])

        redshift = np.array([z])

        redshift_list.append(redshift)
        ra_list.append(ra*np.ones_like(redshift))
        dec_list.append(dec*np.ones_like(redshift))

    # flatten lists
    ra = np.concatenate(ra_list)[::args.stride]
    dec = np.concatenate(dec_list)[::args.stride]
    redshift = np.concatenate(redshift_list)[::args.stride]

    zmax = 7
    max_scale = 1.2

    if args.output:

        ######################
        # z distribution
        ######################

        fig = plt.figure(figsize=(14,8))
        ax = plt.subplot(111, polar=True)

        ax.plot(ra, redshift, marker='.', markersize=1, lw=0)

        ax.set_rmax(max_scale*zmax)
        ax.set_rlabel_position(90)
        ax.grid(True)
        fig.savefig(args.output+'flat.png')

        #######################
        # ra, dec distribution
        #######################

        fig = plt.figure(figsize=(14,8))
        ax = plt.subplot(111, projection='aitoff')

        # matplotlib expects azimuthal angle to be in range [-pi,+pi]
        ra_shifted = ra
        ra_shifted[ra > np.pi] -= 2. * np.pi
        ax.plot(ra_shifted, dec, marker='.', markersize=1, lw=0)

        # plot galactic plane
        galactic_l = np.linspace(0, 2*np.pi, 100)
        galactic_plane = SkyCoord(l=galactic_l*u.radian, b=np.zeros_like(galactic_l)*u.radian, frame='galactic').icrs
        galactic_ra_rad = galactic_plane.ra.radian
        galactic_ra_rad[galactic_ra_rad > np.pi] -= 2. * np.pi
        ax.plot(galactic_ra_rad, galactic_plane.dec.radian, lw=0, marker='.')

        # make pretty
        tick_labels = np.array([210, 240, 270, 300, 330, 0, 30, 60, 90, 120, 150])
        ax.set_xticklabels(tick_labels)
        ax.grid(True)
        fig.savefig(args.output+'aitoff.png')

        #######################
        # 3D distribution
        #######################

        fig = plt.figure(figsize=(14,8))
        ax = fig.add_subplot(111, projection='3d')
        #ax.set_aspect('equal')

        # plot points in 3d
        ab_x = redshift * np.cos(ra) * np.sin(np.pi/2-dec)
        ab_y = redshift * np.sin(ra) * np.sin(np.pi/2-dec)
        ab_z = redshift * np.cos(np.pi/2-dec)
        ax.scatter(ab_x, ab_y, ab_z, marker='.', s=.1)

        # 2d projections
        ax.scatter(ab_x, ab_y, zs=-0.17*max_scale*zmax, zdir='z', marker='.', s=.1)
        ax.scatter(ab_y, ab_z, zs=-max_scale*zmax, zdir='x', marker='.', s=.1)
        ax.scatter(ab_x, ab_z, zs=+max_scale*zmax, zdir='y', marker='.', s=.1)

        # draw shells
        def plot_shell(r):
            # 3d 
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi/2, 100)
            x = r * np.outer(np.cos(u), np.sin(v))
            y = r * np.outer(np.sin(u), np.sin(v))
            z = r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='r', alpha=.05, edgecolor='None')
            # 2d
            x = r * np.cos(u)
            y = r * np.sin(u)

            ind = (y > -0.17*max_scale*zmax)
            ax.plot(x, y, zs=-0.17*max_scale*zmax, zdir='z', color='r', alpha=.4, marker='.', lw=0, ms=1)
            ax.plot(x[ind], y[ind], zs=-max_scale*zmax, zdir='x', color='r', alpha=.4, marker='.', lw=0, ms=1)
            ax.plot(x[ind], y[ind], zs=+max_scale*zmax, zdir='y', color='r', alpha=.4, marker='.', lw=0, ms=1)
        for r in np.arange(1, zmax+1, 1):
            plot_shell(r)

        ax.set_xlim3d(-max_scale*zmax, max_scale*zmax)
        ax.set_ylim3d(-max_scale*zmax, max_scale*zmax)
        ax.set_zlim3d(-0.17*max_scale*zmax, max_scale*zmax)
        # set viewing angle
        ax.view_init(args.elev, args.azim)
        fig.savefig(args.output+'3d.png', bbox_inches='tight')


if __name__ == '__main__':
    main()

