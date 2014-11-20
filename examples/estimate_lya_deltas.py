#!/usr/bin/env python
"""
"""

import argparse

import numpy as np
import scipy.stats
import scipy.interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py

import qusp

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="more verbose output")
    parser.add_argument("--forest-min", type=float, default=1040,
        help="wavelength of lya forest min")
    parser.add_argument("--forest-max", type=float, default=1200,
        help="wavelength of lya forest max")
    parser.add_argument("--wave-lya", type=float, default=1216,
        help="wavelength of lya line")
    parser.add_argument("--z-col", type=int, default=1,
        help="redshift column index")
    parser.add_argument("--output", type=str, default="absorber",
        help="output file name")
    parser.add_argument("--continuum", type=str, default="",
        help="continuum to use")
    qusp.Paths.add_args(parser)
    qusp.target.add_args(parser)
    args = parser.parse_args()

    # setup boss data directory path
    paths = qusp.Paths(**qusp.Paths.from_args(args))
    # read target list
    target_list = qusp.target.load_target_list_from_args(args, 
        fields=[('z', float, args.z_col)])

    # define forest range and lya wavelength
    forest_min = qusp.wavelength.Wavelength(args.forest_min)
    forest_max = qusp.wavelength.Wavelength(args.forest_max)
    wave_lya = qusp.wavelength.Wavelength(args.wave_lya)

    # initialize continuum model
    if args.continuum:
        continuum_model = qusp.LinearFitContinuum(args.continuum)
    else:
        continuum_model = qusp.MeanFluxContinuum()

    ##################################################
    # Calculate mean transmission fraction vs redshift
    ##################################################

    absorber_redshifts = []
    absorber_weights = []
    absorber_transmissions = []

    targets_used_list = []
    continua = []
    forests = []

    # loop over targets
    for target, spplate in qusp.target.get_target_plates(target_list, boss_path=paths.boss_path):
        combined = qusp.spectrum.read_combined_spectrum(spplate, target)

        target['ra'] = spplate[5].data['ra'][target['fiber']-1]
        target['dec'] = spplate[5].data['dec'][target['fiber']-1]

        # determine observed frame forest window
        obs_forest_min = forest_min.observed(target['z'])
        obs_forest_max = forest_max.observed(target['z'])

        # trim the combined spectrum to the forest window
        try:
            forest = combined.trim_range(obs_forest_min, obs_forest_max)
        except ValueError, e:
            # skip target if it's forest is not observable
            print e, '(z = %.2f)' % target['z']
            continue

        # look up continuum for this target
        try:
            continuum = continuum_model.get_continuum(target, forest)
        except ValueError, e:
            # skip target if we can't get a continuum for it
            print e, '(target: %s)' % target.to_string()
            continue

        # calculate absorber redshifts and weights
        absorber_z = forest.wavelength/wave_lya - 1
        absorber_weight = forest.ivar.values
        absorber_transmission = forest.flux.values/continuum.values

        # save this absorbers for this target
        absorber_redshifts.append(absorber_z)
        absorber_weights.append(absorber_weight)
        absorber_transmissions.append(absorber_transmission)

        targets_used_list.append(target)
        continua.append(continuum)
        forests.append(forest)

    # flatten lists
    absorber_redshifts = np.concatenate(absorber_redshifts)
    absorber_weights = np.concatenate(absorber_weights)
    absorber_transmissions = np.concatenate(absorber_transmissions)

    if args.verbose:
        print 'Number of absorbers: %d' % absorber_redshifts.shape[0]
        print 'Mean number per target: %.2f' % (absorber_redshifts.shape[0]/len(target_list))
        print 'Mean absorber redshift: %.4f' % np.mean(absorber_redshifts)
        print 'Mean transimission: %.4f' % np.mean(absorber_transmissions)

    ############################################
    # Save absorber redshift distribution figure
    ############################################

    fig = plt.figure(figsize=(8,6))
    plt.hist(absorber_redshifts, weights=absorber_weights, bins=100, linewidth=.1, alpha=.5)
    plt.xlabel(r'Absorber Redshifts')
    plt.grid()
    fig.savefig(args.output+'-redshifts.png', bbox_inches='tight')

    ##################################
    # Build mean transmission function
    ##################################

    zmax = absorber_redshifts.max() #3.5
    zbinsize = .01
    zbins = np.arange(absorber_redshifts.min(), zmax+zbinsize, zbinsize)
    # digitized = np.digitize(absorber_redshifts, zbins)
    # bin_means = [daabsorber_transmissions[digitized == i].mean() for i in range(1, len(zbins))]

    mean_transmission = scipy.stats.binned_statistic(absorber_redshifts, 
        absorber_transmissions, statistic='mean', bins=zbins)[0]
    count = scipy.stats.binned_statistic(absorber_redshifts, 
        absorber_transmissions, statistic='count', bins=zbins)[0]

    zbin_centers = (zbins[:-1]+zbins[1:])/2
    bad_indices = np.isnan(mean_transmission)
    good_indices = np.logical_not(bad_indices)
    mean_transmission_interp = scipy.interpolate.UnivariateSpline(
        zbin_centers[good_indices], mean_transmission[good_indices], w=np.sqrt(count[good_indices]))

    #################################
    # Save mean transmission function
    #################################

    outfile = h5py.File(args.output+'-delta-field.hdf5', 'w')

    meanfrac_outfile = outfile.create_group('mean_transmission_fraction')
    meanfrac_outfile.create_dataset('z', data=zbin_centers[good_indices])
    meanfrac_outfile.create_dataset('f', data=mean_transmission[good_indices])
    meanfrac_outfile.create_dataset('w', data=np.sqrt(count[good_indices]))

    ####################################################
    # Save mean transmission fraction vs redshift figure
    ####################################################

    fig = plt.figure(figsize=(16,6))

    trans_max = 3 #max(3, mean_transmission.max())
    trans_min = -0.5 #min(-0.5, mean_transmission.min())
    trans_bins = np.linspace(trans_min, trans_max, 100+1)

    plt.hist2d(absorber_redshifts, absorber_transmissions, 
        weights=absorber_weights, bins=[zbins, trans_bins], cmap='Greens')
    plt.plot(zbin_centers, mean_transmission, 'b.')
    plt.plot(zbin_centers, mean_transmission_interp(zbin_centers), 'r-')

    plt.xlabel('absorber redshift')
    plt.ylabel('transmission fraction')
    plt.colorbar()

    fig.savefig(args.output+'-transmission.png', bbox_inches='tight')

    #######################
    # Calculate delta field
    #######################

    absorber_deltas = []

    outfile_delta = outfile.create_group('delta_field')
    # loop over targets
    for i, target in enumerate(targets_used_list):
        forest = forests[i] 
        continuum = continua[i]
        absorber_z = forest.wavelength/wave_lya - 1
        # estimate delta
        deltas = forest.flux.values/(continuum.values*mean_transmission_interp(absorber_z)) - 1
        deltas_ivar = forest.ivar.values/(continuum.values*mean_transmission_interp(absorber_z))
        absorber_deltas.append(deltas)

        # save delta field along this line of sight
        outfile_delta_target = outfile_delta.create_group(target.to_string())
        outfile_delta_target.attrs['ra'] = target['ra']
        outfile_delta_target.attrs['dec'] = target['dec']
        outfile_delta_target.attrs['z'] = target['z']
        outfile_delta_target.create_dataset('absorber_z', data=absorber_z)
        outfile_delta_target.create_dataset('absorber_delta', data=deltas)
        outfile_delta_target.create_dataset('absorber_ivar', data=deltas_ivar)

    # flatten lists
    absorber_deltas = np.concatenate(absorber_deltas)

    if args.verbose:
        print 'Delta mean: %.6f' % np.mean(absorber_deltas)
        print 'Delta var: %.6f' % np.var(absorber_deltas)

    ######################################
    # Save delta field distribution figure
    ######################################

    fig = plt.figure(figsize=(8, 6))
    delta_bins = np.linspace(-5, 5, 100+1)

    # import sklearn.mixture
    # model = sklearn.mixture.GMM(2)
    # result = model.fit(absorber_deltas)

    # logprob, responsibilities = result.score_samples(delta_bins)
    # pdf = np.exp(logprob)
    # pdf_individual = responsibilities * pdf[:, np.newaxis]

    # from scipy.stats import norm
    # mean_delta = result.means_[0, 0]
    # var_delta = result.covars_[0, 0]
    # p1 = norm(mean_delta, np.sqrt(var_delta)).pdf(delta_bins)

    plt.hist(absorber_deltas, weights=absorber_weights, bins=delta_bins, normed=True, linewidth=.1, alpha=.5)
    # plt.plot(delta_bins, pdf, '-k')
    # plt.plot(delta_bins, pdf_individual, '--k')
    plt.xlabel(r'Absorber Deltas')
    plt.grid()
    fig.savefig(args.output+'-deltas.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
