#!/usr/bin/env python
"""
A program to filter FITS catalogs into target lists
"""
import argparse

from astropy.io import fits

import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="optional output FITS file to write")
    parser.add_argument(
        "-i", "--input", type=str, default=None,
        help="required input FITS file to read")
    parser.add_argument(
        "-s", "--select", type=str, default=None,
        help=("tbdata selection string, ex: "
              "\"(tbdata['OBJTYPE'] == 'QSO') & (tbdata['Z'] > 2.1)\""))
    parser.add_argument(
        "--save", type=str, default=None,
        help="target list text file to save")
    parser.add_argument(
        "--annotate", type=str, default=None,
        help=("colon separated list of columns to annotate target list with"
              ", ex: \"ra:dec:z\""))
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
    targets = []
    try:
        fields = args.annotate.split(':')
    except AttributeError:
        fields = []
    # extract targets and fields from table data
    for index, row in enumerate(filtered):
        if args.verbose and (0 == ((index+1) % 1000)):
            print '(processing input entry %d)' % (index+1)
        targetstr = '-'.join(
            [str(row[key]) for key in ['PLATE', 'MJD', 'FIBERID']])
        target = qusp.target.Target(target=targetstr)
        for field in fields:
            target[field] = row[field]
        targets.append(target)
    # save target list
    if args.save:
        # save target list as text file
        if args.verbose:
            print "Saving %d final targets to %s" % (len(targets), args.save)
        qusp.target.save_target_list(args.save, targets, fields)
    # save filtered FITS file
    if args.output:
        if args.verbose:
            print "Saving filtered FITS table to %s" % args.output
        hdulist[1].data = filtered
        hdulist.writeto(args.output, clobber=True)

if __name__ == '__main__':
    main()
