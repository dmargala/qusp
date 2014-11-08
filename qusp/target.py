"""
Provides support for working with BOSS targets.

In qusp, a target is identified by a unique plate-mjd-fiber. They are implemented as dictionaries and 
must have at least 'plate', 'mjd', and 'fiber' keys specified. The Target model is designed to be flexible, 
in that other attributes can be added to targets as needed.

Examples
--------

Construct a target from a string identifier::

    target = qusp.target.Target.from_string('plate-mjd-fiber')


Construct a target from a dictionary::

    target = qusp.target.Target({'target':'plate-mjd-fiber'})


Read a target list along with **ra**, **dec**, and **z** columns::

    targets = qusp.target.load_target_list(filename, fields=[('ra', float, 1), ('dec', float, 2), ('z', float, 3)])


Save a target list along with **z** and **sn** fields::

    qusp.target.save_target_list(filename, targets, fields=['z', 'sn'])


Iterate over combined spectra for a list targets::

    for target, spplate in qusp.target.read_target_plates(paths.boss_path, targets):
        combined = qusp.read_combined_spectrum(spplate, target)
        ...

-------------
"""

import numpy as np

from astropy.io import fits
import os

class Target(dict):
    """
    Represents a BOSS target. 

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Raises:
        AssertionError

    """
    def __init__(self, *args, **kwargs):
        super(Target, self).__init__(*args, **kwargs)
        assert 'target' in self, \
            'Target: must have plate-mjd-fiber identifier key'
        plate, mjd, fiber = self['target'].split('-')
        self['plate'] = int(plate)
        self['mjd'] = int(mjd)
        self['fiber'] = int(fiber)

    def to_string(self):
        """
        Returns the standard plate-mjd-fiber string represntation of the target.

        Returns:
            str
        """
        return self['target']

    @classmethod
    def from_string(cls, target_string):
        """
        Returns a Target object constructed from a target string identifier.

        Args:
            target_string (str): a target string identifier.

        Returns:
            :class:`Target` object
        """
        return cls({'target':target_string})


def load_target_list(filename, fields=None, verbose=False):
    """
    Loads a target data from a text file.

    The first column must be plate-mjd-fiber target identifier.
    Use the fields argument to specify additional columns to
    read. Must specify a (name, type, column index) tuple for each field.

    Args:
        filename (str): The filename to load.
        fields (list, optional): A list of columns to read, see example.
            Defaults to None.
        verbose (bool, optional): Whether or not to print verbose output.
            Defaults to False.

    Returns:
        list of :class:`Target` objects.
    """
    if fields is None:
        fields = []
    fields = [('target', 'S15', 0)] + fields
    names, formats, cols = zip(*fields)
    if verbose:
        print 'Reading fields: %s' % (', '.join(names))
    targets = np.genfromtxt(
        filename, dtype={'names':names, 'formats':formats}, usecols=cols)

    return [Target(dict(zip(targets.dtype.names, t))) for t in targets]

def save_target_list(filename, targets, fields=None, verbose=False):
    """
    Writes a list of targets to the provided file.

    By default, only the target plate-mjd-fiber is written to file.
    Use the fields argument to specify additional target fields to save.

    Args:
        filename (str): The filename of the output text file to create.
        targets (:class:`Target`): A list of :class:`Target` objects to save.
        fields (list, optional): A list of :class:`Target` keys to
            annotate output list. Defaults to None.
        verbose (bool, optional): Whether or not to print verbose output.
            Defaults to False.
    """
    keys = ['target']
    if fields is not None:
        keys.extend(fields if type(fields) is list else [fields])
    if verbose:
        print 'Saving targets to %s w/ fields: %s' % (filename, ', '.join(keys))
    with open(filename, 'w') as outfile:
        for target in targets:
            outfile.write(' '.join([str(target[key]) for key in keys])+'\n')

def read_target_plates(boss_path, targets, sort=True, verbose=False):
    """
    A generator that yields (target,spplate) tuples for the provided list of
    targets. With sort=True, the targets will be sorted by plate-mjd-fiber
    reduce the number of io operations.

    Args:
        boss_path (str): path to boss data directory.
        targets (:class:`Target`): list of :class:`Target` objects
            to iterate through.
        sort (bool, optional): Whether or not to sort the provided targets
            by plate-mjd-fiber. Defaults to True.
        verbose (bool, optional): Whether or not to print verbose output.
            Defaults to False.

    Yields:
        The next tuple ``(target, spplate)``, where *target* is a :class:`Target` and *spplate* is the corresponding FITS file containing its coadded spectrum from the list of *targets*.
    """
    if sort:
        targets = sorted(
            targets, key=lambda t: (t['plate'], t['mjd'], t['fiber']))
    currently_opened_filename = None
    for target in targets:
        plate_filename = 'spPlate-%s-%s.fits' % (target['plate'], target['mjd'])
        if plate_filename != currently_opened_filename:
            # load the spectrum file
            if currently_opened_filename is not None:
                spplate.close()
            full_path_to_spplate = os.path.join(
                boss_path, str(target['plate']), plate_filename)
            if verbose:
                print 'Opening plate file %s...' % full_path_to_spplate
            spplate = fits.open(full_path_to_spplate)
            currently_opened_filename = plate_filename
        yield target, spplate
