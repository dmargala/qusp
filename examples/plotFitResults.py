#!/usr/bin/env python

import h5py
import scipy.sparse
import bosslya

import os
import argparse
from astropy.io import fits

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import ast

# Plot the model matrix, add flare to show model parameter regions and lines between targets
def plotModelMatrix(specfits, targetList):
    modelData = specfits['model_data'].value
    modelIndices = specfits['model_indices'].value
    modelIndPtr = specfits['model_indptr'].value
    modelShape = specfits['model_shape'].value
    npixels = specfits['npixels'].value
    
    restWaveCenters = specfits['restWaveCenters'].value
    obsWaveCenters = specfits['obsWaveCenters'].value
    redshifts = specfits['redshifts'].value
    absorption = specfits['absorption'].value
    absorptionMin = specfits['absorption'].attrs['minRestIndex']
    absorptionMax = specfits['absorption'].attrs['maxRestIndex']
    
    nRest = len(restWaveCenters)
    nObs = len(obsWaveCenters)
    nTargets = len(redshifts)
    nAbs = len(absorption)
    
    model = scipy.sparse.csc_matrix((modelData,modelIndices,modelIndPtr), modelShape)
    
    # pick out rows corresponding to targets in target list
    modelRowIndices = []
    for target in targetList:
        modelRowIndices.append(np.arange(np.sum(npixels[:target]),np.sum(npixels[:target+1])))
    m = scipy.sparse.coo_matrix(model[np.concatenate(modelRowIndices),:])
    
    #print model[np.sum(npixels[:targetList[0]])]
    
    plt.plot(m.col, m.row, 's', color='black',ms=1)
    ax = plt.gca()
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    
    # y axis labels
    yTicks = []
    yTickLabels = []
    for i,target in enumerate(targetList):
        yTickLabels.append('z = %.2f' % redshifts[target])
        yTicks.append(np.sum(npixels[targetList][:i]) + npixels[target]/2)
    ax.set_yticks(yTicks)
    ax.set_yticklabels(yTickLabels)
    ax.invert_yaxis()

    # x axis labels
    # xTickLabels = ['Transmission','Continuum','Absorption','Target']
    # xTicks = [nObs/2,nObs+nRest/2,nObs+nRest+nAlpha/2,nObs+nRest+nAlpha+2*nTargets/2]
    # ax.set_xticks(xTicks)
    # ax.set_xticklabels(xTickLabels)
    # ax.xaxis.tick_top()

    # color background regions
    start, stop = (0, nObs)
    ax.axvspan(start, stop, facecolor='blue', alpha=0.3)
    start = stop
    stop += nRest
    ax.axvspan(start, stop, facecolor='green', alpha=0.3)
    ax.axvspan(start+absorptionMin, start+absorptionMax, facecolor='gray', alpha=.2)
    start = stop
    stop += nAbs
    ax.axvspan(start, stop, facecolor='purple', alpha=0.3)
    start = stop
    stop += 2*nTargets
    ax.axvspan(start, stop, facecolor='orange', alpha=0.3)

    # draw lines at quasar breaks
    for i,target in enumerate(targetList):
        ax.axhline(np.sum(npixels[targetList][:i+1]), c='gray', ls='--')
    
# Plot rest frame continuum
def plotContinuum(specfits, normwave=None, dnorm=None, z=None, **kwargs):
    wavelengths = specfits['restWaveCenters'].value
    continuum = specfits['continuum'].value
    plt.plot(wavelengths, continuum, **kwargs)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.xlabel(r'Rest Wavlength $(\AA)$')
    plt.ylabel(r'Continuum')

def drawNormWindow(wavedset):
    normmin = wavedset.attrs['normmin']
    normmax = wavedset.attrs['normmax']
    plt.axvspan(normmin, normmax, facecolor='gray', alpha=0.5)

# Plot observed frame transmission
def plotTransmission(specfits, **kwargs):
    wavelengths = specfits['obsWaveCenters'].value
    transmission = specfits['transmission'].value
    plt.plot(wavelengths, transmission, **kwargs)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.xlabel(r'Observed Wavlength $(\AA)$')
    plt.ylabel(r'Transmission')
    plt.axhline(1,ls='--',c='gray')

def plotNu(specfits):
    soln = specfits['soln']
    targetNames = specfits['targets'].value
    nTargets = len(targetNames)
    nu = soln[-2*nTargets+1::2]
    plt.hist(nu,bins=50,linewidth=.1, alpha=.5)
    plt.xlabel(r'Spectral Tilt $\nu$')
    plt.ylabel(r'Number of Targets')
    # print '%.4g'%np.mean(nu)
    plt.grid()

def plotAbsorption(specfits, **kwargs):
    absorption = specfits['absorption']
    absmin = absorption.attrs['minRestIndex']
    absmax = absorption.attrs['maxRestIndex']
    restWaves = specfits['restWaveCenters'].value[absmin:absmax]
    plt.plot(restWaves,absorption.value, **kwargs)
    plt.axhline(0.0018,c='gray',ls='--')
    plt.xlabel(r'Rest Wavelength $(\AA)$')
    plt.ylabel(r'Absorption Coefficient $a$')
    plt.xlim([restWaves[0],restWaves[-1]])
    plt.grid()
    
def plotAbsorbedContinuum(specfits, z, **kwargs):
    absorption = specfits['absorption']
    absmodelexp = absorption.attrs['absmodelexp']
    amin = absorption.attrs['minRestIndex']
    amax = absorption.attrs['maxRestIndex']
    restWaves = specfits['restWaveCenters'].value[amin:amax+1]
    continuum = specfits['continuum'].value[amin:amax+1]
    absorption = np.concatenate([absorption.value,[0]])
    absorbed = continuum*np.exp(-absorption*((1+z)**absmodelexp))
    plt.plot(restWaves,absorbed,**kwargs)

def plotAmpVsNu(specfits, **kwargs):
    redshifts = specfits['redshifts'].value
    A = specfits['amplitude'].value
    nu = specfits['nu'].value
    plt.scatter(nu, A, c=redshifts, cmap=plt.cm.jet, **kwargs)
    #plt.gca().set_ylim(bottom=0)
    plt.ylabel(r'Amplitude A')
    plt.yscale('log')
    plt.xlabel(r'Spectral Tilt $\nu$')
    cbar = plt.colorbar(label=r'Redshift $z$')
    cbar.solids.set_edgecolor("face")
    cbar.solids.set_rasterized(True) 

def plotRedshiftDist(specfits):
    redshifts = specfits['redshifts'].value
    plt.hist(redshifts, bins=50, linewidth=.1, alpha=.5)
    plt.xlabel(r'Redshift z')
    plt.ylabel(r'Number of Targets')
    plt.grid()

def plotAmpDist(specfits):
    amp = specfits['amplitude'].value
    plt.hist(amp, bins=10**np.linspace(np.log10(min(amp)), np.log10(max(amp)), 50), linewidth=.1, alpha=.5)
    plt.xlabel(r'Amplitude A')
    plt.ylabel(r'Number of Targets')
    plt.xscale('log')
    plt.grid()

def plotSNDist(specfits):
    sn = specfits['sn'].value
    plt.hist(sn, bins=50, linewidth=.1, alpha=.5)
    plt.xlabel(r'Median Signal-to-Noise Ratio')
    plt.ylabel(r'Number of Targets')
    plt.grid()

def plotChiSq(specfits):
    chisq = specfits['chisq'].value
    plt.hist(chisq, bins=50, linewidth=.1, alpha=.5)
    plt.xlabel(r'ChiSq')
    plt.ylabel(r'Number of Targets')
    plt.grid()

def plotTarget(target, fitsPath):
    plate, mjd, fiber = target['target'].split('-')
    plateFileName = 'spPlate-%s-%s.fits' % (plate, mjd)
    fullName = os.path.join(fitsPath,plate,plateFileName)
    spPlate = fits.open(fullName)
    combined = bosslya.readCombinedSpectrum(spPlate, int(fiber))
    x = combined.wavelength
    y = combined.flux
    plt.plot(x, y, c='b',lw=.5)
    plt.ylabel(r'Flux $(10^{-17} erg/cm^2/s/\AA)$')
    ymax = 1.2*np.percentile(y,99)
    ymin = min(0,1.2*np.percentile(y,1))
    plt.ylim([ymin,ymax])
    spPlate.close()
      
def plotFitTarget(specfits, targetList, fitsPath):
    modelData = specfits['model_data'].value
    modelIndices = specfits['model_indices'].value
    modelIndPtr = specfits['model_indptr'].value
    modelShape = specfits['model_shape'].value
    redshifts = specfits['redshifts'].value
    npixels = specfits['npixels'].value
    targets = specfits['targets'].value
    restWaveCenters = specfits['restWaveCenters'].value
    obsWaveCenters = specfits['obsWaveCenters'].value
    beta = specfits['soln'].value
    aoffset = len(obsWaveCenters)+len(restWaveCenters)
    naparams = len(specfits['absorption'].value)

    model = scipy.sparse.csc_matrix((modelData,modelIndices,modelIndPtr), modelShape)
    
    for j, targetIndex in enumerate(targetList):
        plt.subplot(len(targetList),1,j+1)
        ax = plt.gca()
        # pick out rows corresponding to targets in target list
        modelRowIndices = np.arange(np.sum(npixels[:targetIndex]),np.sum(npixels[:targetIndex+1]))
        
        # remove absorption
        mlil = model[modelRowIndices,:].tolil()
        mlil[:,aoffset:aoffset+naparams] = 0
        
        # convert matrix
        m = scipy.sparse.coo_matrix(mlil)

        # Look up target name and redshift
        target = ast.literal_eval(targets[targetIndex])
        z = redshifts[targetIndex]
        # Draw observation
        plotTarget(target, fitsPath)
        if j+1 == len(targetList):
            plt.xlabel(r'Observed Wavelength $(\AA)$')
        # Draw predication
        wave = obsWaveCenters[m.col[m.col < len(obsWaveCenters)]]
        pred = np.exp(m.dot(beta))/(1+redshifts[targetIndex])
        plt.plot(wave, pred, c='red', marker='', ls='-', lw=1)

        ylim0 = plt.gca().get_ylim()
        ylim = [ylim0[0],max(1.2*max(pred),ylim0[1])]
        plt.ylim(ylim)
        
        ax2 = ax.twinx()
        ax2.set_ylabel(r'%s'%target['target'])
        plt.tick_params(axis='y',labelright='off')
        plt.sca(ax)
        plt.xlim([obsWaveCenters[0], obsWaveCenters[-1]])
        plt.grid()

        bosslya.wavelength.drawLines((1+z)*np.array(bosslya.wavelength.QuasarEmissionLines), bosslya.wavelength.QuasarEmissionLabels, 
            0.89,-0.1, c='orange', alpha=.5)
        bosslya.wavelength.drawLines(bosslya.wavelength.SkyLineList, bosslya.wavelength.SkyLabels, 
            0.01, 0.1, c='magenta', alpha=.5)

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-i","--input", type=str, default=None,
        help="fitspec results file")
    parser.add_argument("-o","--output", type=str, default=None,
        help="output filename base")
    ## BOSS data
    parser.add_argument("--boss-root", type=str, default=None,
        help="path to root directory containing BOSS data (ex: /data/boss)")
    parser.add_argument("--boss-version", type=str, default="v5_7_0",
        help="boss pipeline version tag")
    ## Plot options
    parser.add_argument("--examples", nargs='+', type=int,
        help="plot example spectra")
    args = parser.parse_args()

    mpl.rcParams['font.size'] = 16

    # set up paths
    boss_root = args.boss_root if args.boss_root is not None else os.getenv('BOSS_ROOT', None)
    boss_version = args.boss_version if args.boss_version is not None else os.getenv('BOSS_VERSION', None)
    if boss_root is None or boss_version is None:
        raise RuntimeError('Must speciy --boss-(root|version) or env var BOSS_(ROOT|VERSION)')
    fitsPath = os.path.join(boss_root, boss_version)

    # Import specfits results
    ndefault = h5py.File(args.input)

    ### Target sample plots

    # redshift distribution
    fig = plt.figure(figsize=(8,6))
    plotRedshiftDist(ndefault)
    fig.savefig('%s-redshift.png'%args.output, bbox_inches='tight')
    
    # S/N distribution
    fig = plt.figure(figsize=(8,6))
    plotSNDist(ndefault)
    fig.savefig('%s-sn.png'%args.output, bbox_inches='tight')

    ### Visualize Fit Results
    # Draw Continuum Model
    fig = plt.figure(figsize=(20,8))
    plotContinuum(ndefault,c='black')
    drawNormWindow(ndefault['continuum'])
    bosslya.wavelength.drawLines(bosslya.wavelength.QuasarEmissionLines, bosslya.wavelength.QuasarEmissionLabels, 
        0.89,-0.1, c='orange', alpha=.5)

    # plt.xticks(np.arange(900, 2900, 200))
    plt.grid()
    fig.savefig('%s-continuum.png'%args.output, bbox_inches='tight')

    # Draw Transmission Model
    fig = plt.figure(figsize=(20,8))
    bosslya.wavelength.drawLines(bosslya.wavelength.BallmerLines, bosslya.wavelength.BallmerLabels, 
        0.89,-0.1, c='green', alpha=.5)
    bosslya.wavelength.drawLines(bosslya.wavelength.SkyLineList, bosslya.wavelength.SkyLabels, 
        0.01, 0.1, c='magenta', alpha=.5)
    plotTransmission(ndefault,c='black')
    # plt.xticks(np.arange(3600, 9000, 400))
    plt.ylim([.9,1.1])
    plt.grid()
    fig.savefig('%s-transmission.png'%args.output, bbox_inches='tight')

    # Plot Absorption Model
    fig = plt.figure(figsize=(8,6))
    plotAbsorption(ndefault, c='black')
    bosslya.wavelength.drawLines(bosslya.wavelength.QuasarEmissionLines, bosslya.wavelength.QuasarEmissionLabels, 
        0.01,0.1, c='orange', alpha=.5)
    plt.ylim([0,.004])
    fig.savefig('%s-absorption.png'%args.output, bbox_inches='tight')

    # Plot Spectral Tilt Indices
    fig = plt.figure(figsize=(8,6))
    plotNu(ndefault)
    fig.savefig('%s-nu.png'%args.output, bbox_inches='tight')

    # Compare Amplitudes to Spectral Tilt
    fig = plt.figure(figsize=(8,6))
    plotAmpVsNu(ndefault, marker='o', alpha=1, s=15, linewidth=0)
    plt.grid()
    plt.tight_layout()
    fig.savefig('%s-nuVsA.png'%args.output, bbox_inches='tight')

    # Plot amplitude distribution
    fig = plt.figure(figsize=(8,6))
    plotAmpDist(ndefault)
    fig.savefig('%s-amplitude.png'%args.output, bbox_inches='tight')

    # Plot ChiSq distribution
    fig = plt.figure(figsize=(8,6))
    plotChiSq(ndefault)
    fig.savefig('%s-chisq.png'%args.output, bbox_inches='tight')

    ##### Plot Individual Spectra

    # Create a list of targets to visualize
    if args.examples is not None:
        targetList = args.examples

        ## visualize Model Matrix
        fig = plt.figure(figsize=(15,1*len(targetList)))
        plotModelMatrix(ndefault, targetList)
        fig.savefig('%s-matrix.png'%args.output, bbox_inches='tight')

        ## plot example spectra
        fig = plt.figure(figsize=(15,4*len(targetList)))
        plotFitTarget(ndefault,targetList,fitsPath)
        fig.savefig('%s-examples.png'%args.output, bbox_inches='tight')

        ## worst spectra
        # chisq = ndefault['chisq'].value
        # worstList = sorted(zip(chisq,range(0,len(chisq))),reverse=True)
        # targetList = [value[1] for value in worstList[:nTargets]]

        # ## all spectra
        # numPerPage = 5
        # maxTargets = 5
        # for i in range(0,maxTargets,numPerPage):
        #     n = min(numPerPage,maxTargets-i)
        #     targetList = range(i:i+n)
        #     fig = plt.figure(figsize=(15,4*numPerPage))
        #     plotFitTarget(ndefault,targetList)


if __name__ == '__main__':
    main()

