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
    parser.add_argument("--nzbins", type=float, default=100,
        help = "number of redshift bins")
    parser.add_argument("--norm", action="store_true",
        help = "normalize spectra using window: 1280 +/- 10 Ang")
    args = parser.parse_args()

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)

    if boss_root is None or boss_version is None:
        print 'Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)'
        exit(1)

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
    nzbins = args.nzbins
    zmin = args.zmin
    zmax = args.zmax

    stack = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    counts = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    wstack = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    weight = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    stacksq = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    stackvar = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    wstacksq = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    wstackvar = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    sn = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)

    sqrtwflux = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    sqrtw = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    wfluxsq = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)

    pullmean = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)
    pullrms = numpy.zeros(shape=(arraySize,nzbins), dtype=numpy.float64)

    specialz = int((2.4 - zmin)/(zmax - zmin)*nzbins)
    nonforestflux = []
    forestflux = []

    # work on targets
    plateFileName = None
    for targetIndex in range(ntargets):
        if targetIndex < firstTarget or (endTarget > firstTarget and targetIndex >= endTarget):
            continue
        target = targets[targetIndex]

        # calculate z bin
        if target.z < zmin or target.z > zmax:
            continue
        zbin = int((target.z - zmin)/(zmax - zmin)*nzbins)
        if zbin >= nzbins:
            print 'woh! zbin out of range!'
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
            print 'woh! sprectrum out of range!'
            continue

        # normalize spectrum using flux window: 1280 +/- 10 Ang
        if args.norm:
            obslo = (1+target.z)*1270
            obshi = (1+target.z)*1290
            normlo = int(getFiducialWavelengthRatio(obslo)) - offset
            normhi = int(getFiducialWavelengthRatio(obshi)) - offset
            norm = 0
            normweight = 0
            for pixel in range(normlo,normhi+1):
                if ivar[pixel] > 0:
                    norm += ivar[pixel]*flux[pixel]
                    normweight += ivar[pixel]
            if normweight == 0:
                continue
            norm /= normweight
            flux /= norm
            ivar *= norm*norm

        # add spectrum to stack
        pixelSlice = slice(offset,offset+numPixels)
        stack[pixelSlice,zbin] += flux
        stacksq[pixelSlice,zbin] += flux*flux
        counts[pixelSlice,zbin] += numpy.ones(numPixels)

        if zbin == specialz:
            forestpixel = int(getFiducialWavelengthRatio((1+target.z)*1120)) - offset
            forestflux.append(flux[forestpixel])
            nonforestpixel = int(getFiducialWavelengthRatio((1+target.z)*1500)) - offset
            nonforestflux.append(flux[nonforestpixel])

        wflux = flux*ivar
        wstack[pixelSlice,zbin] += wflux
        wstacksq[pixelSlice,zbin] += wflux*wflux
        weight[pixelSlice,zbin] += ivar

        isigma = numpy.sqrt(ivar)
        sqrtwflux[pixelSlice,zbin] += flux*isigma
        sqrtw[pixelSlice,zbin] += isigma

        wfluxsq[pixelSlice,zbin] += wflux*flux


    # divide by weights/number of entries
    stackSlice = numpy.nonzero(counts)
    stack[stackSlice] /= counts[stackSlice]

    weightSlice = numpy.nonzero(weight)
    wstack[weightSlice] /= weight[weightSlice]

    pullmean[stackSlice] = (sqrtwflux[stackSlice] - wstack[stackSlice]*sqrtw[stackSlice])/counts[stackSlice]
    pullrms[stackSlice] = numpy.sqrt((wfluxsq[stackSlice] - wstack[stackSlice]*weight[stackSlice])/counts[stackSlice])

    sn = stack*numpy.sqrt(weight)

    # save the stacked spectrum matrix
    print 'Saving stack to %s' % outfilename

    outfile.create_dataset('stack', data=stack)
    outfile.create_dataset('wstack', data=wstack)
    outfile.create_dataset('counts', data=counts)
    outfile.create_dataset('weights', data=weight)
    outfile.create_dataset('sn', data=sn)
    outfile.create_dataset('pullmean', data=stackvar)
    outfile.create_dataset('pullrms', data=wstackvar)

    outfile.create_dataset('forest', data=forestflux)
    outfile.create_dataset('nonforest', data=nonforestflux)

    outfile.close()


if __name__ == '__main__':
    main()