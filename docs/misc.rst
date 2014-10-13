Misc
====


Data transfers
--------------

Copy spPlates from darkmatter to hpc:

.. code-block:: bash 

    $ rsync -avz --prune-empty-dirs --include '*/' --include 'spPlate*.fits' --exclude '*' -e ssh dmargala@darkmatter.ps.uci.edu:/data/boss/v5_7_0 /share/dm/all/data/boss/

Copy spAll from darkmatter to hpc:

.. code-block:: bash

	$ scp dmargala@darkmatter.ps.uci.edu:/data/boss/spAll-v5_7_0.fits /share/dm/all/data/boss/

Create target list from lists of plates:

.. code-block:: bash

	for plate in $(cat ~/blue-plates.txt); \
	do \
	examples/filter.py -i /share/dm/all/data/boss/spAll-v5_7_0.fits \
		--select "(tbdata['plate'] == $plate) & (tbdata['objtype'] == 'QSO') & (tbdata['zwarning'] == 0) & (tbdata['z'] > .5)" \
		--save systematics/$plate.txt --annotate 'ra:dec:z' --verbose; \
	done


	