filter
======

A program to select entries from FITS catalogs and create target lists.

::

    usage: filter.py [-h] [--verbose] [-o OUTPUT] [-i INPUT] [-s SELECT]
                     [--save SAVE] [--annotate ANNOTATE]

optional arguments:
  -h, --help            show this help message and exit
  --verbose             print verbose output (default: False)
  -o OUTPUT, --output OUTPUT
                        optional output FITS file to write (default: None)
  -i INPUT, --input INPUT
                        required input FITS file to read (default: None)
  -s SELECT, --select SELECT
                        tbdata selection string, ex: "(tbdata['OBJTYPE'] ==
                        'QSO') & (tbdata['Z'] > 2.1)" (default: None)
  --save SAVE           target list text file to save (default: None)
  --annotate ANNOTATE   colon separated list of columns to annotate target
                        list with, ex: "ra:dec:z" (default: None)
