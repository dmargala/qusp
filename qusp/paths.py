"""
Provides a class to handle paths to boss data directory.
"""
import os
import inspect

class Paths(object):
    """
    Handles a path to a boss data directory.
    """
    def __init__(self, boss_root=None, boss_version=None):
        """
        Initializes a path to a boss data directory.

        Args:
            boss_root (str, optional): The root boss directory path. Defaults to None, in which case, 
                the environment variable BOSS_ROOT is expected to specify the path to the root boss directory.
            boss_version (str, optional): The boss version directory name. Defaults to None, in which case, 
                the environment variable BOSS_VERSION is expected to specify the boss version directory name.

        """
        self.boss_root = boss_root if boss_root is not None else os.getenv('BOSS_ROOT', None)
        if boss_root is None:
            raise RuntimeError('Must speciy --boss-root or env var BOSS_ROOT')
        self.boss_version = boss_version if boss_version is not None else os.getenv('BOSS_VERSION', None)
        if boss_version is None:
            raise RuntimeError('Must speciy --boss-version or env var BOSS_VERSION')
        # construct path to boss data directory
        self.boss_path = os.path.join(self.boss_root, self.boss_version)
        assert os.path.isdir(self.boss_path), 'boss data dir does not exist'

    @staticmethod
    def addArgs(parser):
        """
        Add arguments to the provided command-line parser.

        Args:
            parser (argparse.ArgumentParser): an argument parser
        """
        parser.add_argument("--boss-root", type=str, default=None,
            help = "path to root directory containing BOSS data (ex: /data/boss)")
        parser.add_argument("--boss-version", type=str, default='v5_7_0',
            help = "boss pipeline version tag (ex: v5_7_0)")

    @staticmethod
    def fromArgs(args):
        """
        Returns a dictionary of constructor parameter values based on the parsed args provided.

        Args:
            args (argparse.Namespace): argparse argument namespace

        Returns:
            a dictionary of :class:`Paths` constructor parameter values
        """
        # Look up the named Paths constructor parameters.
        pnames = (inspect.getargspec(Paths.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        argsDict = vars(args)
        # Return a dictionary of constructor parameters provided in args.
        return { key:argsDict[key] for key in (set(pnames) & set(argsDict)) }