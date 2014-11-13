Welcome to qusp's documentation!
================================

A package for working with **qu**\ asar **sp**\ ectra.

Filter targets from spAll:

.. code-block:: bash

    $ examples/filter.py -i spAll-v5_7_0.fits --select "(['OBJTYPE'] == 'QSO') & (['Z'] > 2.1)" --annotate 'ra:dec:z' --verbose --save quasars.txt

The target module provides support for reading target lists:

.. code-block:: python

    targets = qusp.target.load_target_list('quasars.txt', fields=[('ra', float, 1), ('dec', float, 2), ('z', float, 3)])

Use the paths module to manage paths:

.. code-block:: python

    paths = qusp.paths.Paths(boss_root='/Users/daniel/data/boss', boss_version='v5_7_0')

Now we're ready to work with spectra. Here is an example recipe for calculating the median signal to noise in the lya forest for quasar targets:

.. code-block:: python

    # rest frame lya forest limits
    rest_wave_min = qusp.wavelength.Wavelength(1040)
    rest_wave_max = qusp.wavelength.Wavelength(1190)
    for target, spplate in qusp.target.read_target_plates(paths.boss_path, targets):
        # load the combined spectrum for the current target
        combined = qusp.read_combined_spectrum(spplate, target)
        # determine the observed frame forest window
        obs_wave_min = rest_wave_min.observed(target['z'])
        obs_wave_max = rest_wave_max.observed(target['z'])
        # get the median sn in the forest window
        forest_sn = combined.median_signal_to_noise(obs_wave_min, obs_wave_max)


User Documentation
------------------

Package API and how to use the example programs included in this package.

.. toctree::
    :maxdepth: 2

    user
    src/qusp
    src/examples/programs

Developer Documentation
-----------------------

Info on how to build this documentation and other miscellany

.. toctree::
    :maxdepth: 2

    profiling
    docs
    misc


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

