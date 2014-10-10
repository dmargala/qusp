Documentation
=============

Build
-----

Requires sphinx extensions to build:

.. code-block:: bash

    $ pip install sphinx_bootstrap_theme
    $ pip install sphinxcontrib-napoleon
    $ pip install sphinxcontrib-programoutput

Build instructions:

.. code-block:: bash

    $ cd docs
    $ make html
    $ open _build/html/index.html


Initial setup
-------------

.. code-block:: bash

    $ mkdir docs
    $ cd docs
    $ sphinx-quickstart

Edit config.py and specify sys.path to top level:

.. code-block:: python

    sys.path.insert(0, os.path.abspath('..'))

.. code-block:: bash

    $ sphinx-apidoc -o src ../qusp --separate