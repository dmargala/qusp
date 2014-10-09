#!/usr/bin/env python
import argparse
import random

import numpy as np

import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-o", "--output", type=str, default=None,
        help="hdf5 output filename")
    parser.add_argument("--save-model", action="store_true",
        help="specify to save raw data of sparse matrix model")
    parser.add_argument("--save-chisq", action="store_true",
        help="specify to save per obs chisq values")
    ## targets to fit
    parser.add_argument("-i", "--input", type=str, default=None,
        help="target list")
    parser.add_argument("-n", "--ntargets", type=int, default=0,
        help="number of targets to use, 0 for all")
    parser.add_argument("--random", action="store_true",
        help="use a random selection of input targets")
    parser.add_argument("--seed", type=int, default=42,
        help="rng seed")
    # fit options
    parser.add_argument("--sklearn", action="store_true",
        help="use sklearn linear regression instead of scipy lstsq")
    # scipy specifc options
    parser.add_argument("--max-iter", type=int, default=100,
        help="max number of iterations to use in lsqr")
    parser.add_argument("--atol", type=float, default=1e-4,
        help="a stopping tolerance")
    parser.add_argument("--btol", type=float, default=1e-8,
        help="b stopping tolerance")
    # input data columns
    parser.add_argument("--z-col", type=int, default=3,
        help="redshift column of input targetlist")
    parser.add_argument("--sn-col", type=int, default=None,
        help="sn column of input targetlist")
    parser.add_argument("--norm-col", type=int, default=None,
        help="norm param column of input targetlist")
    parser.add_argument("--tilt-col", type=int, default=None,
        help="tilt param column of input targetlist")
    parser.add_argument("--fix-norm", action="store_true",
        help="fix norm param")
    parser.add_argument("--fix-tilt", action="store_true",
        help="fix tilt param")
    qusp.Paths.add_args(parser)
    qusp.ContinuumModel.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))

    # read target data
    fields = [('z', float, args.z_col)]
    if args.norm_col is not None:
        fields.append(('amp', float, args.norm_col))
    if args.tilt_col is not None:
        fields.append(('nu', float, args.tilt_col))
    if args.sn_col is not None:
        fields.append(('sn', float, args.sn_col))
    targets = qusp.target.load_target_list(
        args.input, fields, verbose=args.verbose)

    # use the first n targets or a random sample
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)
    if args.random:
        random.seed(args.seed)
        targets = random.sample(targets, ntargets)
    else:
        targets = targets[:ntargets]

    # Initialize model
    model = qusp.ContinuumModel(**qusp.ContinuumModel.from_args(args))

    # Add observations to model
    model_targets = []
    npixels = []
    if args.verbose:
        print '... adding observations to fit ...\n'
    target_plate_generator = qusp.target.read_target_plates(paths.boss_path, targets)
    for target, spplate in target_plate_generator:
        # read this target's combined spectrum
        combined = qusp.read_combined_spectrum(spplate, target)
        wavelength = combined.wavelength
        ivar = combined.ivar
        flux = combined.flux

        # fix quasar spectrum normalization
        if args.fix_norm:
            if not hasattr(target, 'nu'):
                # estimate quasar normalization
                norm = combined.mean_flux(args.restnormmin*(1+target['z']),
                                          args.restnormmax*(1+target['z']))
                if norm <= 0:
                    continue
                # restframe amplitude
                target['amp'] = norm*(1+target['z'])
        # fix spectal tilt
        if args.fix_tilt:
            if not hasattr(target, 'nu'):
                target['nu'] = 0

        # Add this observation to our model
        npixels_added = model.add_observation(
            target, flux, wavelength, ivar, unweighted=args.unweighted)
        if npixels_added > 0:
            model_targets.append(target)
            npixels.append(npixels_added)

    # Add constraints
    if args.restnormmax > args.restnormmin:
        model.add_rest_constraint(
            0, args.restnormmin, args.restnormmax, args.restnormweight)
    if args.obsnormmax > args.obsnormmin:
        model.add_obs_constraint(
            0, args.obsnormmin, args.obsnormmax, args.obsnormweight)
    if args.tiltweight > 0:
        model.add_tilt_constraint(args.tiltweight)

    if args.verbose:
        print ''

    # Construct the model
    model_matrix, model_y = model.get_model()

    # perform fit
    if args.sklearn:
        from sklearn import linear_model
        regr = linear_model.LinearRegression(fit_intercept=False)
        if args.verbose:
            print ('... performing fit using '
                   'sklearn.linear_model.LinearRegression ...\n')
        regr.fit(model_matrix, model_y)
        soln = regr.coef_
    else:
        import scipy.sparse.linalg
        if args.verbose:
            print '... performing fit using scipy.sparse.linalg.lsqr ...\n'
        lsqr_soln = scipy.sparse.linalg.lsqr(
            model_matrix, model_y, show=args.verbose, iter_lim=args.max_iter,
            atol=args.atol, btol=args.btol)
        soln = lsqr_soln[0]

    chisq = model.get_chisq(soln)

    if args.verbose:
        print 'chisq (nModelParams,nConstraints): %.2g (%d,%d)' % (
            chisq, model.model.shape[1], model.model_nconstraints)
        print 'reduced chisq: %.2g' % (
            chisq/(model.model.shape[1]-model.model_nconstraints))

    # Save HDF5 file with results
    outfile = model.save(args.output+'.hdf5', soln, args, args.save_model, args.save_chisq)

    outfile.create_dataset('npixels', data=npixels)
    outfile.create_dataset(
        'targets', data=[target['target'] for target in model_targets])
    outfile.create_dataset(
        'redshifts', data=[target['z'] for target in model_targets])

    try:
        mediansn = [target['sn'] for target in model_targets]
    except KeyError:
        mediansn = np.zeros(len(model_targets))

    outfile.create_dataset('sn', data=mediansn)
    outfile.close()

    # Save target list text file
    results = model.get_results(soln)
    for index, target in enumerate(model_targets):
        target['amp'] = results['amplitude'][index]
        target['nu'] = results['nu'][index]
    qusp.target.save_target_list(
        args.output+'.txt', model_targets, ['z', 'amp', 'nu'],
        verbose=args.verbose)

if __name__ == '__main__':
    main()
