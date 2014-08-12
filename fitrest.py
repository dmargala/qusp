#!/usr/bin/env python
import argparse

import numpy
import h5py

import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

import plotstack
import matplotlib.pyplot as plt

def main():

    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--input", type=str, default=None,
        help="stacked spectrum file")
    parser.add_argument("-o","--output", type=str, default='output/lsqr',
        help="prepend output files with specified string")
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("--max-iter", type=int, default=100,
        help="max number of iterations to use in lsqr")
    parser.add_argument("--atol", type=float, default=1e-2,
        help="a stopping tolerance")
    parser.add_argument("--btol", type=float, default=1e-8,
        help="b stopping tolerance")
    parser.add_argument("--csr", action="store_true",
        help="convert sparse matrix to csr instead of csc format")
    parser.add_argument("--rest-min", type=float, default=0,
        help="minimum rest wavelength to include")
    parser.add_argument("--rest-max", type=float, default=7000,
        help="maximum rest wavelength to include")
    parser.add_argument("--obs-min", type=float, default=3450,
        help="minimum obs wavelength to include")
    parser.add_argument("--obs-max", type=float, default=10650,
        help="maximum obs wavelength to include")
    args = parser.parse_args()

    # Read input data
    infile = h5py.File(args.input,'r')

    hists2d = infile['hists2d']

    # Open output file
    outfile = h5py.File(args.output+'.hdf5','w')

    # Convert pixel edges to pixel centers and trim input data while we're at its
    xBinEdges = hists2d['xbinedges']
    obsMinIndex = numpy.argmax(xBinEdges.value > args.obs_min)
    obsMaxIndex = numpy.argmax(xBinEdges.value > args.obs_max)

    if obsMaxIndex <= obsMinIndex:
        obsMaxIndex = xBinEdges.len()
    obsWaveSlice = slice(obsMinIndex,obsMaxIndex)
    
    obsWaveEdges = outfile.create_dataset('xbinedges', data=xBinEdges[obsWaveSlice])
    obsWaveEdges.attrs['label'] = 'Observed Wavelength $(\AA)$'

    obsWaveCenters = (obsWaveEdges[:-1] + obsWaveEdges[1:])/2
    obsNPixels = len(obsWaveCenters)

    yBinEdges = hists2d['ybinedges']
    restMinIndex = numpy.argmax(yBinEdges.value > args.rest_min)
    restMaxIndex = numpy.argmax(yBinEdges.value > args.rest_max)

    if restMaxIndex <= restMinIndex:
        restMaxIndex = yBinEdges.len()
    restWaveSlice = slice(restMinIndex,restMaxIndex)
    
    restWaveEdges = outfile.create_dataset('ybinedges', data=yBinEdges[restWaveSlice])
    restWaveEdges.attrs['label'] = 'Rest Wavelength $(\AA)$'

    restWaveCenters = (restWaveEdges[:-1] + restWaveEdges[1:])/2
    restNPixels = len(restWaveCenters)

    print 'Input data dimensions: (%d,%d)' % (xBinEdges.len()-1,yBinEdges.len()-1)
    print 'Dimensions after trimming: (%d,%d)' % (obsNPixels, restNPixels)

    flux = hists2d['wfluxmean'][obsWaveSlice,restWaveSlice]
    counts = hists2d['counts'][obsWaveSlice,restWaveSlice]

    # We only want pixels that have data and positive flux
    # since we are working with logF
    iGood = numpy.logical_and(counts > 0, flux > 0)
    iBad = numpy.logical_or(counts <= 0, flux <= 0)

    logFlux = numpy.log10(flux[iGood])
    logFluxNPixels = len(logFlux)

    # Build sparse matrix representation of model
    modelNPixels = obsNPixels + restNPixels

    iObs, iRest = numpy.indices((obsNPixels, restNPixels))
    iObsGood = iObs[iGood]
    iRestGood = iRest[iGood]
    iArray = numpy.arange(0,logFluxNPixels)

    Arows = numpy.concatenate([iArray,iArray])
    Acols = numpy.concatenate([iObsGood,obsNPixels+iRestGood])

    A = scipy.sparse.coo_matrix((numpy.ones(2*logFluxNPixels),(Arows,Acols)), 
        shape=(logFluxNPixels,modelNPixels), dtype=numpy.int8)
    A = A.tocsr() if args.csr else A.tocsc()

    print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)' % (A.data.nbytes, A.indptr.nbytes, A.indices.nbytes)

    # Perform least squares iteration
    soln = scipy.sparse.linalg.lsqr(A, logFlux, show=args.verbose, iter_lim=args.max_iter, atol=args.atol, btol=args.btol)

    # Construct rest and obs frame model components
    obsModelPixels = 10**soln[0][:obsNPixels]
    restModelPixels = 10**soln[0][obsNPixels:obsNPixels+restNPixels]

    obsModel = scipy.interpolate.UnivariateSpline(obsWaveCenters, obsModelPixels, s=0)
    restModel = scipy.interpolate.UnivariateSpline(restWaveCenters, restModelPixels, s=0)

    # Calculate residuals
    model = numpy.outer(obsModel(obsWaveCenters),restModel(restWaveCenters))
    res = numpy.zeros(shape=flux.shape)
    res[iGood] = model[iGood] - flux[iGood]

    # Save HDF5 file with results
    grp = outfile.create_group('T')
    dset = grp.create_dataset('y', data=obsModelPixels)
    dset = grp.create_dataset('x', data=obsWaveCenters)

    grp = outfile.create_group('C')
    dset = grp.create_dataset('y', data=restModelPixels)
    dset = grp.create_dataset('x', data=restWaveCenters)

    # Save 2D plots
    saveName = args.output+'-data.png'
    plotstack.plothist2d(flux, saveName, 
        obsWaveEdges, restWaveEdges, 'Data',
        vmin=0, vmax=numpy.percentile(flux,99))

    saveName = args.output+'-model.png'
    plotstack.plothist2d(model, saveName, 
        obsWaveEdges, restWaveEdges, 'Model',
        vmin=0, vmax=numpy.percentile(flux,99))

    resAbsMax = max(numpy.percentile(res,99),numpy.percentile(-res,99))
    saveName = args.output+'-res.png'
    plotstack.plothist2d(res, saveName, 
        obsWaveEdges, restWaveEdges, 'Residuals', 
        vmin=-resAbsMax, vmax=resAbsMax, cmap=plt.get_cmap('bwr'))

    # Save 1D plots
    fig = plt.figure(figsize=(10,6))

    plt.plot(restWaveCenters, restModelPixels, '+')
    plt.xlim([restWaveCenters[0],restWaveCenters[-1]])
    plt.xlabel('Rest Wavelength $\lambda_{rest}$')
    plt.ylabel('$C(\lambda_{rest})$')

    fig.savefig(args.output+'-cont.png', bbox_inches='tight')

    fig = plt.figure(figsize=(20,6))

    plt.axhline(y=1, xmin=0, xmax=1, color='gray', linestyle='--')
    plt.plot(obsWaveCenters, obsModelPixels, '+')
    plt.xlim([obsWaveCenters[0],obsWaveCenters[-1]])
    plt.xlabel('Observed Wavelength $\lambda_{obs}$')
    plt.ylabel('$T(\lambda_{obs})$')
    plt.ylim([0, 2])

    fig.savefig(args.output+'-trans.png', bbox_inches='tight')

    outfile.close()

if __name__ == '__main__':
    main()