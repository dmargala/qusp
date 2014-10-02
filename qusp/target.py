"""
Provides support for working with BOSS targets.
"""

import numpy as np

from astropy.io import fits
import os

class Target(dict):
    def __init__(self,*arg,**kw):
        """
        Initializes a Target object. Parses the plate-mjd-fiber identifier
        and adds separate fields to the dictionary.
        """
        super(Target, self).__init__(*arg, **kw)
        assert 'target' in self, 'Target: must have plate-mjd-fiber identifier key'
        plate,mjd,fiber = self['target'].split('-')
        self['plate'] = int(plate)
        self['mjd'] = int(mjd)
        self['fiber'] = int(fiber)
    def toString(self):
        """
        Returns the standard plate-mjd-fiber string represntation of the target.
        """
        return self['target']

def loadTargetData(filename, fields=[], verbose=False):
    """
    Loads a target data from a text file. The first column must be plate-mjd-fiber
    target identifier. Use the fields argument to specify additional columns to
    read. Must specify a (name, type, column index) tuple for each field.

    For example, to read a target list along with ra, dec, and z columns:
        loadTargetData(<filename>, fields=[('ra',float,1),('dec',float,2),('z',float,3)])
    """
    fields = [('target','S15',0)] + fields
    names, formats, cols = zip(*fields)
    if verbose:
        print 'Reading fields: %s' % (', '.join(names))
    targetData = np.genfromtxt(filename,dtype={'names':names,'formats':formats},usecols=cols)

    return [Target(dict(zip(targetData.dtype.names,target))) for target in targetData]

def saveTargetData(filename, targets, fields=[], verbose=False):
    """
    Writes a list of targets to the provided file. By default, only the
    target plate-mjd-fiber is written to file. Use the fields argument
    to specify additional target fields to save.
    """
    keys = ['target']
    keys.extend(fields if type(fields) is list else [fields])
    if verbose:
        print 'Saving fields: %s' % (', '.join(keys))
    with open(filename, 'w') as outfile:
        for target in targets:
            outfile.write(' '.join([str(target[key]) for key in keys])+'\n')

def readTargetPlates(boss_path, targets, sort=True, verbose=False):
    """
    A generator that yields (target,spPlate) tuples for the provided list of 
    targets. With sort=True, the targets will be sorted by plate-mjd-fiber 
    reduce the number of io operations.
    """
    if sort:
        targets = sorted(targets, key=lambda target: target['target'])
    currentlyOpened = None
    for target in targets:
        plateFileName = 'spPlate-%s-%s.fits' % (target['plate'], target['mjd'])
        if plateFileName != currentlyOpened:
            # load the spectrum file
            if currentlyOpened is not None:
                spPlate.close()
            fullName = os.path.join(boss_path,str(target['plate']),plateFileName)
            if verbose:
               print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)
            currentlyOpened = plateFileName
        yield target, spPlate
