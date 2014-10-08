#!/usr/bin/env python

import h5py
import scipy.sparse
import qusp

import os
import argparse
from astropy.io import fits

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.interpolate

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
def plotContinuum(specfits, **kwargs):
    wavelengths = specfits['restWaveCenters'].value
    continuum = specfits['continuum'].value
    plt.plot(wavelengths, continuum, **kwargs)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.xlabel(r'Rest Wavlength $(\AA)$')
    plt.ylabel(r'Continuum')

# Plot observed frame transmission
def plotTransmission(specfits, **kwargs):
    wavelengths = specfits['obsWaveCenters'].value
    transmission = specfits['transmission'].value
    plt.plot(wavelengths, transmission, **kwargs)
    plt.xlim([wavelengths[0], wavelengths[-1]])
    plt.xlabel(r'Observed Wavlength $(\AA)$')
    plt.ylabel(r'Transmission')
    plt.axhline(1,ls='--',c='gray')

def drawNormWindow(wavedset):
    normmin = wavedset.attrs['normmin']
    normmax = wavedset.attrs['normmax']
    plt.axvspan(normmin, normmax, facecolor='gray', alpha=0.5)

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

def plotSpectrum(spectrum, **kwargs):
    plt.plot(spectrum.wavelength, spectrum.flux, **kwargs)
    plt.xlabel(r'Observed Wavelength $(\AA)$')
    plt.ylabel(r'Flux $(10^{-17} erg/cm^2/s/\AA)$')
    ymax = 1.2*np.percentile(spectrum.flux,99)
    ymin = min(0,1.2*np.percentile(spectrum.flux,1))
    plt.ylim([ymin,ymax])
      
def plotTargets(specfits, targetIndices, boss_path):
    # construct list of Targets from specified indices
    targets = specfits['targets'].value
    redshifts = specfits['redshifts'].value
    amp = specfits['amplitude'].value
    nu = specfits['nu'].value    
    mytargets = []
    for i in targetIndices:
        try:
            target = qusp.target.Target.from_string(targets[i])
        except ValueError:
            # for backwards compatibility...
            target = qusp.target.Target(ast.literal_eval(targets[i]))
        target['z'] = redshifts[i]
        target['nu'] = nu[i]
        target['amp'] = amp[i]
        mytargets.append(target)
    ntargets = len(mytargets)

    # build continumm and transmission model
    restWaveCenters = specfits['restWaveCenters'].value
    obsWaveCenters = specfits['obsWaveCenters'].value
    continuum = specfits['continuum'].value
    transmission = specfits['transmission'].value
    T = scipy.interpolate.UnivariateSpline(obsWaveCenters, transmission, s=0)
    C = scipy.interpolate.UnivariateSpline(restWaveCenters, continuum, s=0)

    tiltwave = specfits['nu'].attrs['tiltwave']
    def plotPrediction(target, **kwargs):
        z = target['z']
        amp = target['amp']
        nu = target['nu']
        redshiftedWaves = obsWaveCenters/(1+z)
        pred = amp/(1+z)*(redshiftedWaves/tiltwave)**nu*T(obsWaveCenters)*C(redshiftedWaves)
        plt.plot(obsWaveCenters, pred, **kwargs)
        # set axes ranges
        # ylim0 = plt.gca().get_ylim()
        # ylim = [ylim0[0],min(1.2*max(pred),ylim0[1])]
        # plt.ylim(ylim)
        plt.xlim([obsWaveCenters[0], obsWaveCenters[-1]])

    subplotIndex = 1
    for target, spPlate in qusp.target.read_target_plates(boss_path, mytargets):    
        plt.subplot(ntargets,1,subplotIndex)
        subplotIndex += 1
        ax = plt.gca()
        # draw observed spectrum
        combined = qusp.read_combined_spectrum(spPlate, target)
        plotSpectrum(combined, c='blue', lw=.5)
        # draw predication
        plotPrediction(target, c='red', marker='', ls='-', lw=1)
        quasar_lines = qusp.wavelength.load_wavelengths('quasar')
        for line in quasar_lines:
            line *= (1+target['z'])
        # draw emission lines
        qusp.wavelength.draw_lines(quasar_lines, 0.89,-0.1, c='orange', alpha=.5)
        qusp.wavelength.draw_lines(
            qusp.wavelength.load_wavelengths('calcium'), 0.01, 0.1, c='blue', alpha=.5)
        qusp.wavelength.draw_lines(
            qusp.wavelength.load_wavelengths('sky',ignore_labels=True), 0.01, 0.1, c='magenta', alpha=.3)
        # only keep xaxis label of the bottom/last plot
        if subplotIndex <= ntargets:
            plt.xlabel(r'')
        # add horizontal grid lines
        plt.grid(axis='y')
        # add target label to right side
        ax2 = ax.twinx()
        ax2.set_ylabel(r'%s'%target['target'])
        plt.tick_params(axis='y',labelright='off',right='off')

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-i","--input", type=str, default=None,
        help="fitspec results file")
    parser.add_argument("-o","--output", type=str, default=None,
        help="output filename base")
    ## Plot options
    parser.add_argument("--examples", nargs='+', type=int,
        help="plot example spectra")
    parser.add_argument("--save-model", action="store_true",
        help="save example model matrix")
    ##
    parser.add_argument("--force-y", action="store_true",
        help="force y limit ranges to nominal values")
    qusp.paths.Paths.add_args(parser)
    args = parser.parse_args()

    paths = qusp.paths.Paths(**qusp.paths.Paths.from_args(args))

    mpl.rcParams['font.size'] = 16

    # Import specfits results
    specfits = h5py.File(args.input)

    ##### Target sample plots

    # redshift distribution
    fig = plt.figure(figsize=(8,6))
    redshifts = specfits['redshifts'].value
    plt.hist(redshifts, bins=50, linewidth=.1, alpha=.5)
    plt.xlabel(r'Redshift z')
    plt.ylabel(r'Number of Targets')
    plt.grid()
    fig.savefig('%s-redshift.png'%args.output, bbox_inches='tight')
    
    # S/N distribution
    fig = plt.figure(figsize=(8,6))
    sn = specfits['sn'].value
    plt.hist(sn, bins=50, linewidth=.1, alpha=.5)
    plt.xlabel(r'Median Signal-to-Noise Ratio')
    plt.ylabel(r'Number of Targets')
    plt.grid()
    fig.savefig('%s-sn.png'%args.output, bbox_inches='tight')

    ### Visualize Fit Results
    # Draw Continuum Model
    fig = plt.figure(figsize=(20,8))
    plotContinuum(specfits,c='black')
    drawNormWindow(specfits['continuum'])
    qusp.wavelength.draw_lines(
        qusp.wavelength.load_wavelengths('quasar'), 0.89,-0.1, c='orange', alpha=.5)
    if args.force_y:
        plt.ylim([0,5])
    plt.grid(axis='y')
    fig.savefig('%s-continuum.png'%args.output, bbox_inches='tight')

    # Draw Transmission Model
    fig = plt.figure(figsize=(20,8))
    plotTransmission(specfits,c='black')
    qusp.wavelength.draw_lines(
        qusp.wavelength.load_wavelengths('ballmer'), 0.89,-0.1, c='green', alpha=.5)
    qusp.wavelength.draw_lines(
        qusp.wavelength.load_wavelengths('calcium'), 0.01, 0.1, c='blue', alpha=.5)
    qusp.wavelength.draw_lines(
        qusp.wavelength.load_wavelengths('sky',ignore_labels=True), 0.01, 0.1, c='magenta', alpha=.3)
    if args.force_y:
        plt.ylim([.9,1.1])
    plt.grid(axis='y')
    fig.savefig('%s-transmission.png'%args.output, bbox_inches='tight')

    # Plot Absorption Model
    fig = plt.figure(figsize=(8,6))
    plotAbsorption(specfits, c='black')
    qusp.wavelength.draw_lines(
        qusp.wavelength.load_wavelengths('quasar'), 0.01,0.1, c='orange', alpha=.5)
    if args.force_y:
        plt.ylim([-0.0005,.004])
    fig.savefig('%s-absorption.png'%args.output, bbox_inches='tight')

    # Plot Spectral Tilt Indices
    fig = plt.figure(figsize=(8,6))
    nu = specfits['nu'].value
    plt.hist(nu,bins=50,linewidth=.1, alpha=.5)
    plt.xlabel(r'Spectral Tilt $\nu$')
    plt.ylabel(r'Number of Targets')
    plt.grid()
    fig.savefig('%s-nu.png'%args.output, bbox_inches='tight')

    # Plot amplitude distribution
    fig = plt.figure(figsize=(8,6))
    amp = specfits['amplitude'].value
    plt.hist(amp, bins=10**np.linspace(np.log10(min(amp)), np.log10(max(amp)), 50), 
             linewidth=.1, alpha=.5)
    plt.xlabel(r'Amplitude A')
    plt.ylabel(r'Number of Targets')
    plt.xscale('log')
    plt.grid()
    fig.savefig('%s-amplitude.png'%args.output, bbox_inches='tight')

    # Compare Amplitudes to Spectral Tilt
    fig = plt.figure(figsize=(8,6))
    plt.scatter(nu, amp, c=redshifts, cmap=plt.cm.jet, marker='o', alpha=1, s=15, linewidth=0)
    plt.xlabel(r'Spectral Tilt $\nu$')
    plt.ylabel(r'Amplitude A')
    plt.yscale('log')
    cbar = plt.colorbar(label=r'Redshift $z$')
    cbar.solids.set_edgecolor("face")
    cbar.solids.set_rasterized(True)     
    plt.grid()
    fig.savefig('%s-nuVsA.png'%args.output, bbox_inches='tight')

    # Plot ChiSq distribution
    fig = plt.figure(figsize=(8,6))
    chisq = specfits['chisq'].value
    plt.hist(chisq, bins=50, linewidth=.1, alpha=.5)
    plt.xlabel(r'ChiSq')
    plt.ylabel(r'Number of Targets')
    plt.grid()
    fig.savefig('%s-chisq.png'%args.output, bbox_inches='tight')

    ##### Plot Example Spectra

    # Create a list of targets to visualize
    if args.examples is not None:
        targetIndices = args.examples

        ## plot example spectra
        fig = plt.figure(figsize=(15,4*len(targetIndices)))
        plotTargets(specfits, targetIndices, paths.boss_path)
        fig.savefig('%s-examples.png'%args.output, bbox_inches='tight')

        ## visualize Model Matrix
        if args.save_model:
            fig = plt.figure(figsize=(15,1*len(targetIndices)))
            plotModelMatrix(specfits, targetIndices)
            fig.savefig('%s-matrix.png'%args.output, bbox_inches='tight')

        ## worst spectra
        # chisq = specfits['chisq'].value
        # worstList = sorted(zip(chisq,range(0,len(chisq))),reverse=True)
        # targetList = [value[1] for value in worstList[:nTargets]]

        # ## all spectra
        # numPerPage = 5
        # maxTargets = 5
        # for i in range(0,maxTargets,numPerPage):
        #     n = min(numPerPage,maxTargets-i)
        #     targetList = range(i:i+n)
        #     fig = plt.figure(figsize=(15,4*numPerPage))
        #     plotFitTarget(specfits,targetList)


if __name__ == '__main__':
    main()

