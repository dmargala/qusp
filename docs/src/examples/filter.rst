filter
======

A program to select entries from FITS catalogs and create target lists.

Filter targets from spAll:

.. code-block:: bash

    $ examples/filter.py -i spAll-v5_7_0.fits --select "(['OBJTYPE'] == 'QSO') & (['Z'] > 2.1)" --annotate 'ra:dec:z' --verbose --save quasars.txt
