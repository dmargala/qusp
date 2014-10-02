#!/usr/bin/env python
import argparse
import os

import numpy as np
import h5py
from astropy.io import fits

import matplotlib.pyplot as plt

import bosslya

import desimodel.simulate

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-o","--output", type=str, default=None,
        help="hdf5 output filename")
    ## BOSS data
    parser.add_argument("--boss-root", type=str, default=None,
        help="path to root directory containing BOSS data (ex: /data/boss)")
    parser.add_argument("--boss-version", type=str, default="v5_7_0",
        help="boss pipeline version tag")
    ## targets to fit
    parser.add_argument("-i","--input", type=str, default=None,
        help="target list")
    parser.add_argument("-n","--ntargets", type=int, default=0,
        help="number of targets to use, 0 for all")
    parser.add_argument("--random", action="store_true",
        help="use a random selection of input targets")
    parser.add_argument("--seed", type=int, default=42,
        help="rng seed")
    parser.add_argument("--save-targets", action="store_true",
        help="save individual target spectrum plots")
    parser.add_argument("--catalog", type=str, default="spAll-v5_7_0.fits",
        help="catalog to use for psfmag info")
    parser.add_argument("--magkey", type=str, default="PSFMAG",
        help="catalog magnitude keyword")
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    fitsPath = os.path.join(boss_root, boss_version)

    dr12q = fits.open(os.path.join(boss_root,args.catalog))

    fitResults = h5py.File(args.input)

    redshifts = fitResults['redshifts'].value
    targets = fitResults['targets'].value

    restWaveCenters = fitResults['restWaveCenters'].value
    obsWaveCenters = fitResults['obsWaveCenters'].value

    T = fitResults['T'].value
    C = fitResults['C'].value

    absorption = fitResults['alpha']
    beta = absorption.attrs['beta']
    amin = absorption.attrs['minRestIndex']
    amax = absorption.attrs['maxRestIndex']

    amplitudes = fitResults['A'].value
    tiltindices = fitResults['nu'].value
    nuwave = fitResults['nu'].attrs['nuwave']

    # read target list
    # targets = bosslya.readTargetList(args.input,[('ra',float),('dec',float),('z',float),('thingid',int),('sn',float)])
    ntargets = args.ntargets if args.ntargets > 0 else len(targets)

    # use the first n targets or a random sample
    if args.random:
        np.random.seed(args.seed)
        targets = [targets[i] for i in np.random.randint(len(targets), size=ntargets)]
    else:
        targets = targets[:ntargets]

    # we want to open the spPlate files in plate-mjd order
    #targets = sorted(targets)

    if args.verbose: 
        print 'Read %d targets from %s' % (ntargets,args.input)

    mags = {key:[list(),list(),list(),list()] for key in 'ugriz'}

    # Add observations to fitter
    plateFileName = None
    fitTargets = []
    npixels = []
    for i, targetstr in enumerate(targets):
        target = bosslya.Target.fromString(targetstr)
        # load the spectrum file
        if plateFileName != 'spPlate-%s-%s.fits' % (target.plate, target.mjd):
            if plateFileName:
                spPlate.close()
            plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
            fullName = os.path.join(fitsPath,str(target.plate),plateFileName)
            # if args.verbose:
            #    print 'Opening plate file %s...' % fullName
            spPlate = fits.open(fullName)

        # catalog magnitudes
        catalogIndex = np.where(np.logical_and(
            dr12q[1].data['PLATE'] == target.plate, np.logical_and(
            dr12q[1].data['MJD'] == target.mjd,
            dr12q[1].data['FIBERID'] == target.fiber)))

        try:
            psfmags = dr12q[1].data[catalogIndex][args.magkey][0]
        except IndexError:
            continue

        for key,mag in zip('ugriz',psfmags):
            mags[key][0].append(mag)

        # read this target's combined spectrum
        combined = bosslya.readCombinedSpectrum(spPlate, target.fiber)
        wave = combined.wavelength
        ivar = combined.ivar
        flux = combined.flux

        # Synthetic magnitude
        D = desimodel.simulate.SpectralFluxDensity(wave, flux)
        for key,value in D.getABMagnitudes().iteritems():
            mags[key][1].append(value)

        # Remove transmission model magnitude
        DoverT = desimodel.simulate.SpectralFluxDensity(obsWaveCenters,D(obsWaveCenters)/T)
        for key,value in DoverT.getABMagnitudes().iteritems():
            mags[key][2].append(value)

        # Fit result magnitude
        # A = amplitudes[i]
        # nu = tiltindices[i]
        # z = redshifts[i]

        # frest = A*(restWaveCenters/nuwave)**nu*C
        # frest[amin:amax] *= np.exp(-absorption.value*((1+z)**beta))

        # fgal = desimodel.simulate.SpectralFluxDensity(restWaveCenters, frest, extrapolatedValue=0).createRedshifted(z)
        # fobs = desimodel.simulate.SpectralFluxDensity(obsWaveCenters, T*fgal(obsWaveCenters))

        # for key,value in fobs.getABMagnitudes().iteritems():
        #     mags[key][3].append(value)

        if args.save_targets:
            # Draw observed spectrum and prediction
            fig = plt.figure(figsize=(16,4))
            plt.plot(wave,flux,c='b',lw=.5)
            plt.plot(obsWaveCenters,fobs(obsWaveCenters),c='r')

            # Adjust plot range
            plt.xlim([obsWaveCenters[0], obsWaveCenters[-1]])
            ylim0 = plt.gca().get_ylim()
            ylim = [min(0,1.2*np.percentile(flux,1)),1.2*np.percentile(flux,99)]
            plt.ylim(ylim)

            # Annotate with common emission/absorption lines
            bosslya.wavelength.drawLines((1+z)*np.array(bosslya.wavelength.QuasarEmissionLines), bosslya.wavelength.QuasarEmissionLabels, 
                0.89,-0.1, c='orange', alpha=.5)
            bosslya.wavelength.drawLines(bosslya.wavelength.SkyLineList, bosslya.wavelength.SkyLabels, 
                0.01, 0.1, c='magenta', alpha=.5)

            # Label axes
            plt.xlabel(r'Observed Wavelength $(\AA)$')
            plt.ylabel(r'Flux $(10^{-17} erg/cm^2/s/\AA)$')

            # Draw target label
            ax2 = plt.gca().twinx()
            ax2.set_ylabel(r'%s'%str(target))
            plt.tick_params(axis='y',labelright='off')
            plt.grid()

            if not os.path.exists(args.output):
                os.makedirs(args.output)

            fig.savefig('%s/%s.png'%(args.output,str(target)), bbox_inches='tight')
            plt.close()

    for key in mags.keys():
        for i,magtype in enumerate(['D','D/T']):#,r'$A (\lambda/\lambda_{\star})^{\nu} C exp(-\tau)/(1+z)$']):
            fig = plt.figure(figsize=(8,6))
            plt.scatter(mags[key][0],mags[key][1+i], alpha=0.5)
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
            plt.plot(xlim,xlim)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.ylabel(magtype)
            plt.xlabel(args.magkey)
            plt.grid()
            fig.savefig('%s-%s-%s-%s.png'%(args.output,args.magkey,str(i),key), bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()