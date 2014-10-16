#!/usr/bin/env python
"""
A program to filter FITS catalogs into target lists

Basic example:

    examples/filter.py -i spAll-v5_7_0.fits --select "(tbdata['OBJTYPE'] == 'QSO') & (tbdata['Z'] > 2.1)" --annotate 'ra:dec:z' --verbose --save quasars.txt


Advanced example (not very efficient):

    for plate in $(cat ~/blue-plates.txt); do examples/filter.py -i /share/dm/all/data/boss/spAll-v5_7_0.fits --select "(tbdata['plate'] == $plate) & (tbdata['objtype'] == 'QSO') & (tbdata['zwarning'] == 0) & (tbdata['z'] > .5)" --save systematics/$plate.txt --annotate 'ra:dec:z' --verbose; done

"""
import argparse

from astropy.io import fits

import numpy as np

import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-o", "--output", type=str, default=None,
        help="optional output FITS file to write")
    parser.add_argument("-i", "--input", type=str, default=None,
        help="required input FITS file to read")
    parser.add_argument("-s", "--select", type=str, default=None,
        help="tbdata selection string, ex: \"(tbdata['OBJTYPE'] == 'QSO') & (tbdata['Z'] > 2.1)\"")
    parser.add_argument("--save", type=str, default=None,
        help="target list text file to save")
    parser.add_argument("--annotate", type=str, default=None,
        help="colon separated list of columns to annotate target list with, ex: 'ra:dec:z'")
    args = parser.parse_args()

    assert args.input is not None, 'No input file specified'
    # read input FITS file
    hdulist = fits.open(args.input)
    # filter table data
    tbdata = hdulist[1].data
    if args.verbose:
        print 'Read %d entries from %s' % (len(tbdata), args.input)
    if args.select:
        mask = eval(args.select)
        filtered = tbdata[mask]
        if args.verbose:
            print 'Selecting %d rows matching: %s' % (
                len(filtered), args.select)
    else:
        filtered = tbdata
        if args.verbose:
            print 'No selection specified (selecting all entries).'

    # construct list of fields to annotate target list
    try:
        fields = args.annotate.split(':')
    except AttributeError:
        fields = []
    # save target list
    if args.save:
        names = ['plate','mjd','fiberid'] + fields
        cols = [filtered.field(col) for col in names]
        formats = ','.join([filtered[col].dtype.name for col in names])
        trimmed = np.rec.fromarrays(cols, names=names, dtype=formats)
        # save target list as text file
        if args.verbose:
            print "Saving %d final targets to %s" % (len(trimmed), args.save)
        fmt = '%s-%s-%s ' + ' '.join(['%s' for field in fields])
        np.savetxt(args.save, trimmed, fmt=fmt)
    # save filtered FITS file
    if args.output:
        if args.verbose:
            print "Saving filtered FITS table to %s" % args.output
        hdulist[1].data = filtered
        hdulist.writeto(args.output, clobber=True)

if __name__ == '__main__':
    main()
