Misc
====


Data transfers
--------------

Copy spPlates from darkmatter to hpc:

.. code-block:: bash 

    $ rsync -avz --prune-empty-dirs --include '*/' --include 'spPlate*.fits' --exclude '*' -e ssh dmargala@darkmatter.ps.uci.edu:/data/boss/v5_7_0 /share/dm/all/data/boss/