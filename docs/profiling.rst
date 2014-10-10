Profiling
=========

Run program w/ profiler:

.. code-block:: bash

    $ python -m cProfile -o profile.out <program> <args>

View stats in interactive session:

.. code-block:: python

    import pstats
    p = pstats.Stats('profile.out')
    p.sort_stats('time').print_stats(10)

Create call a tree dot graph :

.. code-block:: bash
   
    $ gprof2dot -f pstats profile.out | dot -Tpng -o profile.png

Might need to install ``gprof2dot``:

.. code-block:: bash

    $ pip install gprof2dot