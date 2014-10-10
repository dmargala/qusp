Environment Setup
=================

HPC
---

Only tested using interactive session so far... 

::

    qrsh -q dm

Use a local user install of anaconda for python.

::

    module purge
    export PATH=/data/users/dmargala/anaconda/bin:$PATH
    export PYTHONPATH=/data/users/dmargala/source/qusp

    export BOSS_ROOT=/share/dm/all/data/boss
    export BOSS_VERSION=v5_7_0

Darkmatter
----------

::

    export PYTHONPATH=/home/dmargala/source/qusp

    export BOSS_ROOT=/share/dm/all/data/boss
    export BOSS_VERSION=v5_7_0