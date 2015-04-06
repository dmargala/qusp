"""
Provides a class to manage paths to boss data directories.

Examples
--------

Via standard constructor:

>>> paths = qusp.paths.Paths(boss_root='/Users/daniel/data/boss', boss_version='v5_7_0')

Via environment variables::

    BOSS_ROOT='/Users/daniel/data/boss'
    BOSS_VERSION='v5_7_0'

>>> paths = qusp.paths.Paths()

Via command line arguments::

    ./example --boss-root /Users/daniel/data/boss --boss-version v5_7_0

where the example program looks something like::

    def main():
        parser = argparse.ArgumentParser()
        qusp.Paths.add_args(parser)
        args = parser.parse_args()
        paths = qusp.Paths(**qusp.Paths.from_args(args))

------------

"""
import os
import inspect

class Paths(object):
    """
    Manages a path to a boss data directory.
    """
    def __init__(self, boss_root=None, boss_version=None):
        """
        Args:
            boss_root (str, optional): The root boss directory path.
                Defaults to None, in which case, the environment variable
                `BOSS_ROOT` is expected to specify the path to the root
                boss directory.
            boss_version (str, optional): The boss version directory name.
                Defaults to None, in which case, the environment variable
                `BOSS_VERSION` is expected to specify the boss version
                directory name.

        Raises:
            RuntimeError if either `boss_root` or `boss_version` are not
                specifed by parameters or environment variables.

        """
        if boss_root is not None:
            self.boss_root = boss_root
        else:
            self.boss_root = os.getenv('BOSS_ROOT', None)
        if self.boss_root is None:
            raise RuntimeError(
                'Must speciy --boss-root or env var BOSS_ROOT')
        if boss_version is not None:
            self.boss_version = boss_version
        else:
            self.boss_version = os.getenv('BOSS_VERSION', None)
        if self.boss_version is None:
            raise RuntimeError(
                'Must speciy --boss-version or env var BOSS_VERSION')
        # construct path to boss data directory
        self.boss_path = os.path.join(self.boss_root, self.boss_version)
        assert os.path.isdir(self.boss_path), 'boss data dir does not exist'

    def get_spplate_filename(self, target):
        """
        Returns full path to the specified target's spPlate fits file.

        Args:
            target (:class:`qusp.target.Target`)

        Returns:
            spplate_filename (str)
        """
        plate_filename = 'spPlate-%s-%s.fits' % (target['plate'], target['mjd'])
        return os.path.join(self.boss_path, str(target['plate']), plate_filename)

    def get_spec_filename(self, target, lite=False):
        spec_filename = os.path.join(('%04d' % target['plate']), 'spec-%04d-%5d-%04d.fits' % (target['plate'], target['mjd'], target['fiber']))
        if lite:
            spec_filename = os.path.join('lite', spec_filename)
        return os.path.join(self.boss_path, 'spectra', spec_filename)

    @staticmethod
    def add_args(parser):
        """
        Adds arguments to the provided command-line parser.

        Args:
            parser (argparse.ArgumentParser): an argument parser
        """
        parser.add_argument(
            "--boss-root", type=str, default=None,
            help="path to root directory containing BOSS data (ex: /data/boss)")
        parser.add_argument(
            "--boss-version", type=str, default=None,
            help="boss pipeline version tag (ex: v5_7_0)")

    @staticmethod
    def from_args(args):
        """
        Returns a dictionary of constructor parameter values based on
        the parsed args provided.

        Args:
            args (argparse.Namespace): argparse argument namespace

        Returns:
            a dictionary of :class:`Paths` constructor parameter values
        """
        # Look up the named Paths constructor parameters.
        pnames = (inspect.getargspec(Paths.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        args_dict = vars(args)
        # Return a dictionary of constructor parameters provided in args.
        return {key:args_dict[key] for key in set(pnames) & set(args_dict)}

