Welcome to qusp's documentation!
================================

A package for working with **qu**\ asar **sp**\ ectra.

Here is an example recipe for calculating the median signal to noise in the lya forest for quasar targets:

.. code-block:: python

    # rest frame lya forest limits
    rest_wave_min = qusp.wavelength.Wavelength(1040)
    rest_wave_max = qusp.wavelength.Wavelength(1190)
    for target, spectrum in qusp.target.get_combined_spectra(targets):
        # determine the observed frame forest window
        obs_wave_min = rest_wave_min.observed(target['z'])
        obs_wave_max = rest_wave_max.observed(target['z'])
        # get the median sn in the forest window
        forest_sn = spectrum.median_signal_to_noise(obs_wave_min, obs_wave_max)

The target module provides support for reading target lists:

.. code-block:: python

    # parse target list with ra, dec, and z attributes, example line:
    #   1234-56789-109 87.6 54.3 2.1
    targets = qusp.target.load_target_list('quasars.txt', 
        fields=[('ra', float, 1), ('dec', float, 2), ('z', float, 3)])

The ``filter.py`` utility makes it east to create target lists from a catalog file:

.. code-block:: bash

    $ examples/filter.py -i spAll-v5_7_0.fits \
        --select "(['OBJTYPE'] == 'QSO') & (['Z'] > 2.1)" \
        --annotate 'ra:dec:z' --save quasars.txt --verbose 


User Documentation
------------------

Package API and how to use the example programs included in this package.

.. toctree::
    :maxdepth: 2

    quick
    src/qusp
    src/examples/programs

Developer Documentation
-----------------------

Info on how to build this documentation and other miscellany

.. toctree::
    :maxdepth: 2

    dev


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

