#!/usr/bin/env python
"""
Stacks BOSS spectra
"""

import os
import math
import argparse

import numpy

from astropy.io import fits

import ROOT
import h5py

class Target(object):
    def __init__(self, plate, mjd, fiber):
        self.plate = int(plate)
        self.mjd = int(mjd)
        self.fiber = int(fiber)

    @classmethod
    def fromString(cls, targetString):
        targetString = targetString.strip().split()[0]
        plate, mjd, fiber = targetString.split('-')
        return cls(plate, mjd, fiber);

    def __str__(self):
        return '%d-%d-%d' % (self.plate,self.mjd,self.fiber)

def getFiducialWavelength(pixelIndex):
    return 3500.26*(10**(1e-4*pixelIndex))

def getFiducialWavelengthRatio(lambda1, lambda2=3500.26):
    return 1e4*math.log10(lambda1/lambda2)

def getFiducialPixelIndexOffset(coeff0, coeff1=1e-4):
    if coeff1 != 1e-4:
        return 0
    delta = (math.log10(3500.26)-coeff0)/coeff1
    offset = int(math.floor(delta+0.5))
    if math.fabs(delta-offset) > 0.01:
        return 0
    return -offset

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
    parser.add_argument("--skim", type=str, default="dr12v1",
        help = "name of skim to use")
    parser.add_argument("--out-prefix", type=str, default="root",
        help = "output file prefix")
    parser.add_argument("--verbose", action="store_true",
        help = "more verbose output")
    parser.add_argument("--use-fits", action="store_true",
        help = "read spectra from PLATE/spPlate-PLATE-MJD.fits files, \
        otherwise use skim/plate-PLATE-MJD.root files")
    parser.add_argument("--zmin", type=float, default=2.1,
        help = "minimum quasar redshift to include")
    parser.add_argument("--zmax", type=float, default=3,
        help = "maximum quasar redshift to include")
    parser.add_argument("--restmin", type=float, default=750,
        help = "restframe wavelength minimum")
    parser.add_argument("--restmax", type=float, default=3500,
        help = "restframe wavelength maximum")
    parser.add_argument("--nrestbins", type=float, default=1000,
        help = "number of redshift bins")
    parser.add_argument("--norm", action="store_true",
        help = "normalize spectra using mean flux value in a specified window")
    parser.add_argument("--norm-lo", type=float, default=1270,
        help = "min wavelength for normalization window")
    parser.add_argument("--norm-hi", type=float, default=1290,
        help = "max wavelength for normalization window")
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')

    skimPath = os.path.join(boss_root, 'skim', args.skim)
    fitsPath = os.path.join(boss_root, boss_version)

    # read target list
    targets = []
    with open(args.input,'r') as targetlist:
        for line in targetlist:
            words = line.strip().split()

            target = Target.fromString(line)

            target.z = float(words[3])
            targets.append(target)

    ntargets = len(targets)
    firstTarget = args.first_target
    endTarget = firstTarget + args.ntargets

    print 'Read %d targets (using %d:%d) from %s' % (ntargets,firstTarget,endTarget,args.input)

    # open output files ahead of time to catch potential io errors before processing data
    try:
        outfilename = args.out_prefix+'.hdf5'
        outfile = h5py.File(outfilename, 'w')
    except IOError:
        print 'Failed to open output file: ' % outfilename 

    # initialize stack arrays
    arraySize = 4800
    zmin = args.zmin
    zmax = args.zmax

    restmin = args.restmin
    restmax = args.restmax
    nrestbins = args.nrestbins

    fluxsum = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)
    counts = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)
    wfluxsum = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)
    weights = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)
    weightssq = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)
    fluxsq = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)
    wfluxsq = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)

    sqrtwflux = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)
    sqrtw = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)
    wfluxsq = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float64)

    skipcounter = 0

    # work on targets
    plateFileName = None
    for targetIndex in range(ntargets):
        if targetIndex < firstTarget or (endTarget > firstTarget and targetIndex >= endTarget):
            continue
        target = targets[targetIndex]

        # calculate z bin
        if target.z < zmin or target.z > zmax:
            continue

        # read spectrum
        flux = []
        wflux = []
        ivar = []
        numPixels = 0
        coeff0 = 0

        if args.use_fits:
            # load the spectrum file
            if plateFileName != 'spPlate-%s-%s.fits' % (target.plate, target.mjd):
                plateFileName = 'spPlate-%s-%s.fits' % (target.plate, target.mjd)
                if args.verbose:
                    print 'Opening plate file %s...' % os.path.join(skimPath,plateFileName)
                plateFile = fits.open(os.path.join(fitsPath,str(target.plate),plateFileName)) 
                # when does the file close?

            numPixels = plateFile[0].header['NAXIS1']
            coeff0 = plateFile[0].header['COEFF0']

            index = target.fiber-1
            andmask = plateFile[2].data[index]

            flux = plateFile[0].data[index]
            ivar = plateFile[1].data[index]

            ivar[andmask > 0] = 0

        else:
            # load the spectrum file
            if plateFileName != 'plate-%s-%s.root' % (target.plate, target.mjd):
                plateFileName = 'plate-%s-%s.root' % (target.plate, target.mjd)
                if args.verbose:
                    print 'Opening plate file %s...' % os.path.join(skimPath,plateFileName)
                plateFile = ROOT.TFile(os.path.join(skimPath,plateFileName))
                # when does the file close?
                plateTree = plateFile.Get('skim')

            combined = plateFile.Get('combined_%s' % target.fiber)
            numPixels = combined.GetN()

            xBuffer = combined.GetX()
            wavelength = numpy.frombuffer(xBuffer,count=numPixels)

            coeff0 = math.log10(wavelength[0])

            yBuffer = combined.GetY()
            flux = numpy.frombuffer(yBuffer,count=numPixels)

            yErrBuffer = combined.GetEY()
            fluxErr = numpy.frombuffer(yErrBuffer,count=numPixels)

            ivar = numpy.zeros(numPixels)
            nonzeroEntries = numpy.nonzero(fluxErr)
            ivar[nonzeroEntries] = 1/(fluxErr[nonzeroEntries]**2)

        # determine pixel offset
        offset = getFiducialPixelIndexOffset(coeff0)
        if numPixels + offset > arraySize:
            raise RuntimeError('woh! sprectrum out of range!')

        obswave = (10**coeff0)*numpy.power(10, 1e-4*(numpy.arange(0, numPixels)))
        restwave = obswave/(1+target.z)
        restindices = ((restwave - restmin)/(restmax - restmin)*nrestbins).astype(int)

        # normalize spectrum using flux window: 1280 +/- 10 Ang
        if args.norm:
            obslo = (1+target.z)*args.norm_lo
            obshi = (1+target.z)*args.norm_hi
            normlo = int(getFiducialWavelengthRatio(obslo)) - offset
            normhi = int(getFiducialWavelengthRatio(obshi)) - offset + 1
            # limit norm window to spectrum 
            if normlo < 0:
                normlo = 0
            if normhi > numPixels or args.norm_hi < args.norm_lo:
                normhi = numPixels
            # calucuate mean flux in window
            normSlice = slice(normlo,normhi)
            norm = numpy.sum(ivar[normSlice]*flux[normSlice])
            normweight = numpy.sum(ivar[normSlice])
            if normweight == 0:
                skipcounter += 1
                print 'skipping %s (z=%.2f): 0 ivar in flux normalization range' % (target, target.z)
                continue
            norm /= normweight
            flux /= norm
            ivar *= norm*norm

        # add spectrum to stack
        #pixelSlice = slice(offset,offset+numPixels)
        obsindices = numpy.arange(offset,offset+numPixels)

        r = numpy.logical_or(restindices < 0, restindices >= nrestbins)
        ivar[r] = 0

        i = ivar > 0
        flux[ivar == 0] = 0
        fluxsum[obsindices,restindices] += flux
        # fluxsq[obsindices,restindices] += flux*flux
        counts[obsindices,restindices] += i

        wflux = flux*ivar
        wfluxsum[obsindices,restindices] += wflux
        weights[obsindices,restindices] += ivar
        # weightssq[obsindices,restindices] += ivar*ivar

        # isigma = numpy.sqrt(ivar)
        # sqrtw[obsindices,restindices] += isigma
        # sqrtwflux[obsindices,restindices] += isigma*flux

        # wfluxsq[obsindices,restindices] += wflux*flux

    print 'Skipped %d targets...' % skipcounter

    # divide by weights/number of entries
    fluxmean = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float32)
    c = numpy.nonzero(counts)
    fluxmean[c] = fluxsum[c]/counts[c]

    wfluxmean = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float32)
    w = numpy.nonzero(weights)
    wfluxmean[w] = wfluxsum[w]/weights[w]

    # wwdenom = weights**2 - weightssq
    # ww = numpy.nonzero(wwdenom)
    # wfluxvar = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float32)
    # wfluxvar[ww] = weights[ww]/(wwdenom[ww])*(wfluxsq[ww] - (wfluxmean[ww]**2)*weights[ww])

    # pullmean = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float32)
    # pullvar = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float32)

    # pullmean[w] = (sqrtwflux[w] - wfluxmean[w]*sqrtw[w])/counts[w]
    # pullvar[w] = (wfluxsq[w] - (wfluxmean[w]**2)*weights[w])/counts[w]

    # sn = numpy.zeros(shape=(arraySize,nrestbins), dtype=numpy.float32)
    # sn[c] = numpy.sqrt(wfluxsq[c]/counts[c])

    # save the stacked spectrum matrix
    print 'Saving stack to %s' % outfilename

    grp = outfile.create_group('hists2d')

    xaxis = 3500.26*numpy.power(10, 1e-4*(numpy.arange(0, arraySize+1)-0.5))
    dset = grp.create_dataset('xbinedges', data=xaxis)
    dset.attrs['label'] = 'Observed Wavelength ($\AA$)'

    yaxis = numpy.linspace(restmin,restmax,nrestbins+1,endpoint=True)
    dset = grp.create_dataset('ybinedges', data=yaxis)
    dset.attrs['label'] = 'Rest Wavelength ($\AA$)'

    dset = grp.create_dataset('fluxmean', data=fluxmean)
    dset.attrs['label'] = 'Normalized Flux Mean'# $(10^{-17} erg/cm^2/s/\AA)$'

    dset = grp.create_dataset('wfluxmean', data=wfluxmean)
    dset.attrs['label'] = 'Normalized Weighted Flux Mean'# $(10^{-17} erg/cm^2/s/\AA)$'

    # dset = grp.create_dataset('wfluxvar', data=wfluxvar)
    # dset.attrs['label'] = 'Normalized Weighted Flux Variance'# $(10^{-17} erg/cm^2/s/\AA)^{-2}$'

    dset = grp.create_dataset('counts', data=counts)
    dset.attrs['label'] = 'Counts'

    dset = grp.create_dataset('weights', data=weights)
    dset.attrs['label'] = 'Normalized Weights'# $(10^{-17} erg/cm^2/s/\AA)^{-2}$'

    # dset = grp.create_dataset('sn', data=sn)
    # dset.attrs['label'] = 'Signal to Noise Ratio'

    # dset = grp.create_dataset('pullmean', data=pullmean)
    # dset.attrs['label'] = 'Pull Mean'

    # dset = grp.create_dataset('pullvar', data=pullvar)
    # dset.attrs['label'] = 'Pull Variance'

    outfile.close()


if __name__ == '__main__':
    main()