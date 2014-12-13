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

    targets = qusp.target.load_target_list(filename, 
        fields=[('ra', float, 1), ('dec', float, 2), ('z', float, 3)])


Save a target list along with **z** and **sn** fields::

    qusp.target.save_target_list(filename, targets, fields=['z', 'sn'])


Iterate over combined spectra for a list targets::

    for target, spectrum in qusp.target.get_combined_spectra(targets):
        ...

Iterate over plates for a list of targets::

    for target, spplate in qusp.target.get_target_plates(targets):
        spectrum = qusp.read_combined_spectrum(spplate, target)
        ...

-------------
"""

import numpy as np

from astropy.io import fits
import os

import qusp

class Target(dict):
    """
    Represents a BOSS target. 

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Raises:
        AssertionError: if 'target' key is not specified

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
            plate-mjd-fiber string represntation of target
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

    @classmethod
    def from_plate_mjd_fiber(cls, plate, mjd, fiber):
        """
        Returns a Target object constructed from plate, mjd, fiber.

        Args:
            plate (int): target's plate id
            mjd (int): mjd observation
            fiber (int): target's fiber id

        Returns:
            :class:`Target` object
        """
        target_string = '-'.join([str(field) for field in [plate, mjd, fiber]])
        return cls.from_string(target_string)

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
        print 'Target list: %s' % filename
        print 'Reading fields: %s' % (', '.join(names))
    targets = np.genfromtxt(
        filename, dtype={'names':names, 'formats':formats}, usecols=cols)

    return [Target(dict(zip(targets.dtype.names, t))) for t in targets]

def add_args(parser):
    """
    Adds arguments to the provided command-line parser.

    Args:
        parser (argparse.ArgumentParser): an argument parser
    """
    parser.add_argument(
        "--targets", type=str, default=None,
        help="text file containing target list")
    parser.add_argument(
        "--ntargets", type=int, default=0,
        help="number of targets to use, 0 for all")

def load_target_list_from_args(args, fields=None):
    """
    Loads a target list from a text file specified using the default target arguments.

    Args:
        args (argparse.Namespace): argparse argument namespace
        fields (list, optional): A list of columns to read, see example.
            Defaults to None.

    Returns:
        list of :class:`Target` objects.
    """
    target_list = load_target_list(args.targets, fields=fields)
    # trim target list if requested
    ntargets = args.ntargets if args.ntargets > 0 else len(target_list)
    print 'Using %d targets (out of %d in file)' % (ntargets, len(target_list))
    return target_list[:ntargets]

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

def get_target_plates(targets, boss_path=None, sort=True, verbose=False):
    """
    A generator that yields (target,spplate) tuples for the provided list of
    targets. With sort=True, the targets will be sorted by plate-mjd-fiber to
    reduce the number of io operations.

    Args:
        targets (:class:`Target`): list of :class:`Target` objects
            to iterate through.
        boss_path (str, optional): path to boss data directory. Default is to 
            look this up using env var.
        sort (bool, optional): Whether or not to sort the provided targets
            by plate-mjd-fiber. Defaults to True.
        verbose (bool, optional): Whether or not to print verbose output.
            Defaults to False.

    Yields:
        The next tuple ``(target, spplate)``, where ``target`` is a :class:`Target` and \
        ``spplate`` is the corresponding FITS file containing its coadded spectrum from the list of *targets*.
    """
    if boss_path is None:
        paths = qusp.paths.Paths()
        boss_path = paths.boss_path
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

def get_combined_spectra(targets, boss_path=None, sort=True, verbose=False):
    """
    A generator that yields (target, spectrum) tuples for the provided list of
    targets. With sort=True, the targets will be sorted by plate-mjd-fiber to
    reduce the number of io operations.

    Args:
        targets (:class:`Target`): list of :class:`Target` objects
            to iterate through.
        boss_path (str, optional): path to boss data directory. Default is to 
            look this up using env var.
        sort (bool, optional): Whether or not to sort the provided targets
            by plate-mjd-fiber. Defaults to True.
        verbose (bool, optional): Whether or not to print verbose output.
            Defaults to False.

    Yields:
        The next tuple ``(target, spectrum)``, where ``target`` is a :class:`Target` and \
            ``spectrum`` is a :class:`qusp.spectrum.Spectrum` that corresponds to the \
            target's coadded spectrum.
    """

    for target, spplate in get_target_plates(targets, boss_path=boss_path, sort=sort, verbose=verbose):
        combined = qusp.spectrum.read_combined_spectrum(spplate, target)
        yield target, combined

def get_combined_spectrum(target, paths=None):
    """
    Returns the coadded spectrum of the specified target.

    Args:
        target (:class:`Target`): a target 
        paths (:class:`qusp.paths.Paths`, optional): paths object that knows 
            where the location of the boss data dir.

    Returns:
        Coadded spectrum of the specified target.
    """
    if paths is None:
        paths = qusp.paths.Paths()
    plate_filename = paths.get_spplate_filename(target)
    spplate = fits.open(plate_filename)
    return qusp.spectrum.read_combined_spectrum(spplate, target)

def get_corrected_spectrum(target, tpcorr, paths=None):
    """
    Returns the coadded spectrum of the specified target.

    Args:
        target (:class:`Target`): a target 
        tpcorr (hdf5 File object): hdf5 file with throughput corrections
        paths (:class:`qusp.paths.Paths`, optional): paths object that knows 
            where the location of the boss data dir.

    Returns:
        Coadded spectrum of the specified target.
    """
    from scipy.interpolate import interp1d
    combined = get_corrected_spectrum(target, paths)
    wave = tpcorr['wave'].value
    value = tpcorr['%s/%s/%s' % (target['plate'], target['mjd'], target['fiber'])].value
    correction = interp1d(wave, value, kind='linear', copy=False)
    return combined.create_corrected(correction)

