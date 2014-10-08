fitspec
=======

A progam for performing simultaneous least squares fits of BOSS quasar 
spectra to a universal quasar continuum model with redshift dependent :math:`Ly\alpha` absorption, an observed frame transmission model, and individual quasar parameters.

::

  usage: fitspec.py [-h] [--verbose] [-o OUTPUT] [--save-model] [-i INPUT]
                  [-n NTARGETS] [--random] [--seed SEED] [--sklearn]
                  [--max-iter MAX_ITER] [--atol ATOL] [--btol BTOL]
                  [--z-col Z_COL] [--sn-col SN_COL] [--norm-col NORM_COL]
                  [--tilt-col TILT_COL] [--fix-norm] [--fix-tilt]
                  [--boss-root BOSS_ROOT] [--boss-version BOSS_VERSION]
                  [--obsmin OBSMIN] [--obsmax OBSMAX]
                  [--obsnormmin OBSNORMMIN] [--obsnormmax OBSNORMMAX]
                  [--obsnormweight OBSNORMWEIGHT] [--restmin RESTMIN]
                  [--restmax RESTMAX] [--nrestbins NRESTBINS]
                  [--restnormmin RESTNORMMIN] [--restnormmax RESTNORMMAX]
                  [--restnormweight RESTNORMWEIGHT] [--absmin ABSMIN]
                  [--absmax ABSMAX] [--absmodelexp ABSMODELEXP]
                  [--absscale ABSSCALE] [--tiltwave TILTWAVE]
                  [--tiltweight TILTWEIGHT] [--unweighted]

optional arguments:
  -h, --help            show this help message and exit
  --verbose             print verbose output (default: False)
  -o OUTPUT, --output OUTPUT
                        hdf5 output filename (default: None)
  --save-model          specify to save raw data of sparse matrix model
                        (default: False)
  -i INPUT, --input INPUT
                        target list (default: None)
  -n NTARGETS, --ntargets NTARGETS
                        number of targets to use, 0 for all (default: 0)
  --random              use a random selection of input targets (default:
                        False)
  --seed SEED           rng seed (default: 42)
  --sklearn             use sklearn linear regression instead of scipy lstsq
                        (default: False)
  --max-iter MAX_ITER   max number of iterations to use in lsqr (default: 100)
  --atol ATOL           a stopping tolerance (default: 0.0001)
  --btol BTOL           b stopping tolerance (default: 1e-08)
  --z-col Z_COL         redshift column of input targetlist (default: 3)
  --sn-col SN_COL       sn column of input targetlist (default: None)
  --norm-col NORM_COL   norm param column of input targetlist (default: None)
  --tilt-col TILT_COL   tilt param column of input targetlist (default: None)
  --fix-norm            fix norm param (default: False)
  --fix-tilt            fix tilt param (default: False)
  --boss-root BOSS_ROOT
                        path to root directory containing BOSS data (ex:
                        /data/boss) (default: None)
  --boss-version BOSS_VERSION
                        boss pipeline version tag (ex: v5_7_0) (default:
                        v5_7_0)
  --obsmin OBSMIN       transmission model wavelength minimum (default: 3600)
  --obsmax OBSMAX       transmission model wavelength maximum (default: 10000)
  --obsnormmin OBSNORMMIN
                        obsframe wavelength to normalize at (default: 3600)
  --obsnormmax OBSNORMMAX
                        obsframe window size +/- on each side of obsnorm
                        wavelength (default: 10000)
  --obsnormweight OBSNORMWEIGHT
                        norm constraint weight (default: 10.0)
  --restmin RESTMIN     rest wavelength minimum (default: 900)
  --restmax RESTMAX     rest wavelength maximum (default: 2900)
  --nrestbins NRESTBINS
                        number of restframe bins (default: 1000)
  --restnormmin RESTNORMMIN
                        restframe window normalization minimum (default: 1275)
  --restnormmax RESTNORMMAX
                        restframe window normalization maximum (default: 1285)
  --restnormweight RESTNORMWEIGHT
                        norm constraint weight (default: 1000.0)
  --absmin ABSMIN       absorption min wavelength (rest frame) (default: 900)
  --absmax ABSMAX       absoprtion max wavelength (rest frame) (default: 1400)
  --absmodelexp ABSMODELEXP
                        absorption model (1+z) factor exponent (default: 3.92)
  --absscale ABSSCALE   scale absorption params in fit (default: 1)
  --tiltwave TILTWAVE   spectral tilt pivot wavelength (default: 1280)
  --tiltweight TILTWEIGHT
                        spectral tilt constraint weight (default: 1000.0)
  --unweighted          perform unweighted least squares fit (default: False)

