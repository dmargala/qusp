#!/usr/bin/env python

import h5py
import qusp
import argparse
import numpy as np

import matplotlib.pyplot as plt

import scipy.signal

def plot_stack(stack, **kwargs):
    wavelength = stack['wavelength'].value
    flux_wmean = stack['flux_wmean'].value
    weight_sum = stack['weight_sum'].value
    ntargets = stack.attrs['ntargets']

    ylim = plt.ylim()
    ymax = max(1.2*np.percentile(flux_wmean,90),ylim[1])
    plt.ylim([0,ymax])

    plt.plot(wavelength, flux_wmean, **kwargs)

    #plt.plot(wavelength, 0.1*weight_sum/ntargets, alpha=.5, color='gray')

def plot_stack_ratio(stack1, stack2, **kwargs):
    wavelength = stack1['wavelength'].value
    flux1 = stack1['flux_wmean'].value
    flux2 = stack2['flux_wmean'].value
    
    nonzero = np.nonzero(flux2)

    plt.plot(wavelength[nonzero], flux1[nonzero]/flux2[nonzero], **kwargs)

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plate", type=int, default=None,
        help="plate number")
    parser.add_argument("--prefix", type=str, default=None,
        help="input file name prefix")
    parser.add_argument("--output", type=str, default=None,
        help="output prefix")
    qusp.paths.Paths.add_args(parser)
    args = parser.parse_args()

    stds_v5_7_0 = h5py.File('%sstds-v5_7_0-%s.hdf5' % (args.prefix, args.plate))
    offset_v5_7_0 = h5py.File('%soffset_stds-v5_7_0-%s.hdf5' % (args.prefix, args.plate))
    offset_offset = h5py.File('%soffset_stds-v5_7_0_offset-%s.hdf5' % (args.prefix, args.plate))
    stds_offset = h5py.File('%sstds-v5_7_0_offset-%s.hdf5' % (args.prefix, args.plate))

    # plot stacks

    fig = plt.figure(figsize=(20,6))
        
    plot_stack(stds_v5_7_0, c='red', label='5400 (5400)', marker='.', ms=1, ls='None')
    plot_stack(offset_v5_7_0, c='magenta', label='4000 (5400)', marker = '.' , ms=1, ls='None')

    plot_stack(stds_offset, c='blue', label='5400 (4000)', marker='.', ms=1, ls='None')
    plot_stack(offset_offset, c='cyan', label='4000 (4000)', marker='.', ms=1, ls='None')

    plt.legend(loc=4, markerscale=10, numpoints=1)

    plt.xlim([3500,10500])
    #plt.ylim([0,1.5])

    plt.xlabel(r'Wavelength $(\AA)$')
    plt.ylabel('Stacked Flux')

    plt.grid()

    qusp.wavelength.draw_lines(qusp.wavelength.load_wavelengths('balmer'),0.95,-0.05, c='green', alpha=.5)
    qusp.wavelength.draw_lines(qusp.wavelength.load_wavelengths('calcium'),0.01, 0.1, c='blue', alpha=.5)
    #qusp.wavelength.draw_lines(qusp.wavelength.load_wavelengths('sky', ignore_labels=True), 0.01, 0.1, c='magenta', alpha=.3)

    fig.savefig('%s-stacks-%s.png'%(args.output, args.plate), bbox_inches='tight')

    # plot stack ratio

    fig = plt.figure(figsize=(20,6))
        
    plot_stack_ratio(stds_offset, stds_v5_7_0, c='green', label='5400 (4000) / 5400 (5400)', marker='.', ms=1, ls='None')
    plot_stack_ratio(offset_offset, offset_v5_7_0, c='yellow', label='4000 (4000) / 4000 (5400)', marker='.', ms=1, ls='None')

    plt.legend(loc=4, markerscale=10, numpoints=1)

    plt.xlim([3500,10500])
    plt.ylim([.5,2])

    plt.xlabel(r'Wavelength $(\AA)$')
    plt.ylabel('Stacked Flux Ratio')

    plt.grid()

    qusp.wavelength.draw_lines(qusp.wavelength.load_wavelengths('balmer'),0.95,-0.05, c='green', alpha=.5)
    qusp.wavelength.draw_lines(qusp.wavelength.load_wavelengths('calcium'),0.01, 0.1, c='blue', alpha=.5)
    #qusp.wavelength.draw_lines(qusp.wavelength.load_wavelengths('sky',ignore_labels=True), 0.01, 0.1, c='magenta', alpha=.3)

    fig.savefig('%s-stack-ratios-%s.png'%(args.output, args.plate), bbox_inches='tight')

    wavelength = offset_offset['wavelength'].value
    flux1 = offset_offset['flux_wmean'].value
    flux2 = offset_v5_7_0['flux_wmean'].value
    
    nonzero = np.nonzero(flux2)

    stack_ratio = np.array([wavelength[nonzero], flux1[nonzero]/flux2[nonzero]])
    np.savetxt('%s-stack-ratio-%s.txt'%(args.output, args.plate), stack_ratio.T)



if __name__ == '__main__':
    main()




