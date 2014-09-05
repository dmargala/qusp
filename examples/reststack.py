#!/usr/bin/env python
"""
Stacks BOSS spectra
"""

import os
import argparse

import numpy as np

from astropy.io import fits
import h5py

import bosslya

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--input", type=str, default=None,
        help = "target list")
    parser.add_argument("-n","--ntargets", type=int, default=0,
        help = "number of targets to use, 0 for all")
    parser.add_argument("--first-target", type=int, default=0,
        help = "index of first target to use")
    parser.add_argument("--boss-root", type=str, default=None,
        help = "path to root directory containing BOSS data (ex: /data/boss)")
    parser.add_argument("--boss-version", type=str, default=None,
        help = "boss pipeline version tag (ex: v5_7_0)")
    parser.add_argument("--out-prefix", type=str, default="stack",
        help = "output file prefix")
    parser.add_argument("--verbose", action="store_true",
        help = "more verbose output")
    parser.add_argument("--zmin", type=float, default=2.1,
        help = "minimum quasar redshift to include")
    parser.add_argument("--zmax", type=float, default=3,
        help = "maximum quasar redshift to include")
    parser.add_argument("--nzbins", type=float, default=100,
        help = "number of redshift bins")
    parser.add_argument("--restmin", type=float, default=530,
        help = "restframe wavelength minimum")
    parser.add_argument("--restmax", type=float, default=7000,
        help = "restframe wavelength maximum")
    parser.add_argument("--nrestbins", type=float, default=1000,
        help = "number of redshift bins")
    parser.add_argument("--norm", action="store_true",
        help = "normalize spectra using mean flux value in a specified window")
    parser.add_argument("--norm-lo", type=float, default=1270,
        help = "min wavelength for normalization window")
    parser.add_argument("--norm-hi", type=float, default=1290,
        help = "max wavelength for normalization window")
    parser.add_argument("--resty", action="store_true",
        help = "use rest frame wavelength for y binning")
    parser.add_argument("--compression", type=str, default="gzip",
        help = "compress output file using specified scheme")
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    fitsPath = os.path.join(boss_root, boss_version)

    # read target list
    targets = bosslya.readTargetList(args.input,[('ra',float),('dec',float),('z',float),('thingid',int)])

    ntargets = len(targets)
    firstTarget = args.first_target
    endTarget = firstTarget + args.ntargets

    print 'Read %d targets (using %d:%d) from %s' % (ntargets,firstTarget,endTarget,args.input)

    # initialize stack arrays
    nxbins = 4800
    xbincenters = bosslya.getFiducialWavelength(np.arange(nxbins))

    zmin = args.zmin
    zmax = args.zmax
    nzbins = args.nzbins

    restmin = args.restmin
    restmax = args.restmax
    nrestbins = args.nrestbins

    nybins = nrestbins if args.resty else nzbins
    ymin = restmin if args.resty else zmin
    ymax = restmax if args.resty else zmax

    fluxsum = np.zeros(shape=(nxbins,nybins), dtype=np.float64)
    counts = np.zeros(shape=(nxbins,nybins), dtype=np.float64)
    wfluxsum = np.zeros(shape=(nxbins,nybins), dtype=np.float64)
    weights = np.zeros(shape=(nxbins,nybins), dtype=np.float64)
    weightssq = np.zeros(shape=(nxbins,nybins), dtype=np.float64)
    fluxsq = np.zeros(shape=(nxbins,nybins), dtype=np.float64)
    wfluxsq = np.zeros(shape=(nxbins,nybins), dtype=np.float64)

    sqrtwflux = np.zeros(shape=(nxbins,nybins), dtype=np.float64)
    sqrtw = np.zeros(shape=(nxbins,nybins), dtype=np.float64)
    wfluxsq = np.zeros(shape=(nxbins,nybins), dtype=np.float64)

    skipcounter = 0
    sncounter = 0

    # work on targets
    plateFileName = None
    for targetIndex in range(ntargets):
        if targetIndex < firstTarget or (endTarget > firstTarget and targetIndex >= endTarget):
            continue
        target = targets[targetIndex]

        # calculate z bin
        if target.z < zmin or target.z > zmax:
            continue

        # load the spectrum file
        if plateFileName != 'spPlate-%s-%s.fits' % (target.plate, target.mjd):
            plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
            fullPath = os.path.join(fitsPath,str(target.plate),plateFileName)
            if args.verbose:
                print 'Opening plate file %s...' % fullPath
            spPlate = fits.open(fullPath) 

        # read this target's combined spectrum
        combined = bosslya.readCombinedSpectrum(spPlate, target.fiber)

        # determine pixel offset
        offset = bosslya.getFiducialPixelIndexOffset(np.log10(combined.wavelength[0]))
        if combined.nPixels + offset > nxbins:
            raise RuntimeError('woh! sprectrum out of range!')

        flux = combined.flux
        ivar = combined.ivar
        nPixels = combined.nPixels

        mediansn = combined.getMedianSignalToNoise(combined.wavelength[0],combined.wavelength[-1])
        # print 'Median SN: %f' % mediansn
        if mediansn > args.min_sn:
            sncounter += 1
        else:
            continue

        # normalize spectrum using flux window: 1280 +/- 10 Ang
        if args.norm:
            obslo = (1+target.z)*args.norm_lo
            obshi = (1+target.z)*args.norm_hi
            norm = combined.getMeanFlux(obslo, obshi, ivarWeighting=True)
            if norm == 0:
                skipcounter += 1
                print 'skipping %s (z=%.2f): weighted mean flux in norm range is 0' % (str(target), target.z)
                continue
            flux /= norm
            ivar *= norm*norm
            # print 'Norm (%f-%f): %f' % (args.norm_lo, args.norm_hi, norm)

        if args.resty:
            obswave = xbincenters[slice(offset,offset+nPixels)]
            restwave = obswave/(1+target.z)
            restindices = ((restwave - restmin)/(restmax - restmin)*nrestbins).astype(int)
            validbins = np.logical_and(restindices < nybins, restindices >= 0)
            ivar = ivar[validbins]
            flux = flux[validbins]
            yslice = restindices[validbins]
            xslice = np.arange(offset,offset+nPixels)[validbins]
        else:
            zbin = int((target.z - zmin)/(zmax - zmin)*nzbins)
            if zbin >= nzbins:
                raise RuntimeError('woh! zbin out of range!')
            yslice = zbin
            xslice = slice(offset,offset+nPixels)

        # add spectrum to stack
        i = ivar > 0
        flux[ivar == 0] = 0
        fluxsum[xslice,yslice] += flux
        fluxsq[xslice,yslice] += flux*flux
        counts[xslice,yslice] += i

        wflux = flux*ivar
        wfluxsum[xslice,yslice] += wflux
        weights[xslice,yslice] += ivar
        weightssq[xslice,yslice] += ivar*ivar

        isigma = np.sqrt(ivar)
        sqrtw[xslice,yslice] += isigma
        sqrtwflux[xslice,yslice] += isigma*flux

        wfluxsq[xslice,yslice] += wflux*flux

    print 'Skipped %d targets...' % skipcounter
    print 'Number of targets with median SN > %.2f: %d' % (args.min_sn, sncounter) 

    # divide by weights/number of entries
    fluxmean = np.zeros(shape=(nxbins,nybins), dtype=np.float32)
    c = np.nonzero(counts)
    fluxmean[c] = fluxsum[c]/counts[c]

    wfluxmean = np.zeros(shape=(nxbins,nybins), dtype=np.float32)
    w = np.nonzero(weights)
    wfluxmean[w] = wfluxsum[w]/weights[w]

    wwdenom = weights**2 - weightssq
    ww = np.nonzero(wwdenom)
    wfluxvar = np.zeros(shape=(nxbins,nybins), dtype=np.float32)
    wfluxvar[ww] = weights[ww]/(wwdenom[ww])*(wfluxsq[ww] - (wfluxmean[ww]**2)*weights[ww])

    pullmean = np.zeros(shape=(nxbins,nybins), dtype=np.float32)
    pullvar = np.zeros(shape=(nxbins,nybins), dtype=np.float32)

    pullmean[w] = (sqrtwflux[w] - wfluxmean[w]*sqrtw[w])/counts[w]
    pullvar[w] = (wfluxsq[w] - (wfluxmean[w]**2)*weights[w])/counts[w]

    sn = np.zeros(shape=(nxbins,nybins), dtype=np.float32)
    sn[c] = np.sqrt(wfluxsq[c]/counts[c])

    # save the stacked spectrum matrix
    try:
        outfilename = args.out_prefix+'.hdf5'
        outfile = h5py.File(outfilename, 'w')
    except IOError:
        print 'Failed to open output file: ' % outfilename 
    print 'Saving stack to %s' % outfilename

    grp = outfile.create_group('hists2d')

    def saveDataset(name, data, label, units=None, norm=False):
        dset = grp.create_dataset(name, data=data, compression=args.compression)
        if norm:
            label = 'Normalized %s' % label
        elif units:
            label = '%s %s' % (label,units)
        dset.attrs['label'] = label
        return dset

    xbinedges = bosslya.getFiducialWavelength(np.arange(0, nxbins+1)-0.5) 
    xdset = saveDataset('xbinedges', xbinedges, 'Observed Wavelength', '$(\AA)$')

    if not args.resty:
        xdset.attrs['x2label'] = 'Restframe (z = %.1f) Wavelength ($\AA$)' % zmax
        xdset.attrs['x2min'] = xbinedges[0]/(1+ymax)
        xdset.attrs['x2max'] = xbinedges[-1]/(1+ymax)

    ybinedges = np.linspace(ymin,ymax,nybins+1,endpoint=True)
    ylabel = 'Rest Wavelength' if args.resty else 'Redshift'
    yunits = '$(\AA)$' if not args.resty else '(z)'
    saveDataset('ybinedges', ybinedges, ylabel, yunits)

    saveDataset('fluxmean', fluxmean, 'Flux Mean', '$(10^{-17} erg/cm^2/s/\AA)$', args.norm)
    saveDataset('counts', counts, 'Counts')
    saveDataset('wfluxmean', wfluxmean, 'Weighted Flux Mean', '$(10^{-17} erg/cm^2/s/\AA)$', args.norm)
    saveDataset('weights', weights, 'Weights', '$(10^{-17} erg/cm^2/s/\AA)^{-2}$', args.norm)

    saveDataset('wfluxvar', wfluxvar, 'Flux Variance', '$(10^{-17} erg/cm^2/s/\AA)^{2}$', args.norm)
    saveDataset('sn', sn, 'Signal to Noise Ratio')
    saveDataset('pullmean', pullmean, 'Pull Mean')
    saveDataset('pullvar', pullvar, 'Pull Variance')

    outfile.close()


if __name__ == '__main__':
    main()