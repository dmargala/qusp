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
    parser.add_argument("--sklearn", action="store_true",
        help="use sklearn linear regression")
    parser.add_argument("--beta", type=float, default=3.92,
        help="optical depth power law parameter")
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
    
    obsWaveEdges = outfile.create_dataset('xbinedges', data=xBinEdges[obsMinIndex:obsMaxIndex])
    obsWaveEdges.attrs['label'] = 'Observed Wavelength $(\AA)$'

    obsWaveCenters = (obsWaveEdges[:-1] + obsWaveEdges[1:])/2
    obsNPixels = len(obsWaveCenters)
    obsWaveSlice = slice(obsMinIndex,obsMinIndex+obsNPixels)

    yBinEdges = hists2d['ybinedges']
    restMinIndex = numpy.argmax(yBinEdges.value > args.rest_min)
    restMaxIndex = numpy.argmax(yBinEdges.value > args.rest_max)

    if restMaxIndex <= restMinIndex:
        restMaxIndex = yBinEdges.len()

    restWaveEdges = outfile.create_dataset('ybinedges', data=yBinEdges[restMinIndex:restMaxIndex])
    restWaveEdges.attrs['label'] = 'Rest Wavelength $(\AA)$'

    restWaveCenters = (restWaveEdges[:-1] + restWaveEdges[1:])/2
    restNPixels = len(restWaveCenters)
    restWaveSlice = slice(restMinIndex,restMinIndex+restNPixels)

    print 'Input data dimensions: (%d,%d)' % (xBinEdges.len()-1,yBinEdges.len()-1)
    print 'Dimensions after trimming: (%d,%d)' % (obsNPixels, restNPixels)

    flux = hists2d['wfluxmean'][obsWaveSlice,restWaveSlice]
    counts = hists2d['counts'][obsWaveSlice,restWaveSlice]

    # We only want pixels that have data and positive flux
    # since we are working with logF
    iGood = numpy.logical_and(counts > 0, flux > 0)
    iBad = numpy.logical_or(counts <= 0, flux <= 0)

    logFlux = numpy.log(flux[iGood])
    logFluxNPixels = len(logFlux)

    print 'Entries in data matrix: %d' % logFluxNPixels

    # Build sparse matrix representation of model
    modelNPixels = obsNPixels + 2*restNPixels

    iObs, iRest = numpy.indices((obsNPixels, restNPixels))
    iObsGood = iObs[iGood]
    iRestGood = iRest[iGood]
    iArray = numpy.arange(0,logFluxNPixels)

    Arows = numpy.concatenate([iArray,iArray,iArray])
    Acols = numpy.concatenate([iObsGood,obsNPixels+iRestGood,obsNPixels+restNPixels+iRestGood])

    beta = args.beta
    alphaCoefs = -numpy.power(obsWaveCenters[iObsGood]/(restWaveCenters[iRestGood]),beta)
    Avalues = numpy.concatenate([numpy.ones(2*logFluxNPixels),alphaCoefs])

    A = scipy.sparse.coo_matrix((Avalues,(Arows,Acols)),
        shape=(logFluxNPixels,modelNPixels), dtype=numpy.float32)
    A = A.tocsr() if args.csr else A.tocsc()

    print 'Total nbytes of sparse matrix arrays (data, ptr, indices): (%d,%d,%d)' % (A.data.nbytes, A.indptr.nbytes, A.indices.nbytes)

    # Perform least squares iteration
    if args.sklearn:
        from sklearn import linear_model
        regr = linear_model.LinearRegression()
        regr.fit(A, logFlux)
        coef = regr.coef_
    else:
        soln = scipy.sparse.linalg.lsqr(A, logFlux, show=args.verbose, iter_lim=args.max_iter, atol=args.atol, btol=args.btol)
        coef = soln[0]

    # Construct rest and obs frame model components
    obsModelPixels = numpy.exp(coef[:obsNPixels])
    restModelPixels = numpy.exp(coef[obsNPixels:obsNPixels+restNPixels])
    alphaModelPixels = coef[obsNPixels+restNPixels:]

    obsModel = scipy.interpolate.UnivariateSpline(obsWaveCenters, obsModelPixels, s=0)
    restModel = scipy.interpolate.UnivariateSpline(restWaveCenters, restModelPixels, s=0)
    alphaModel = scipy.interpolate.UnivariateSpline(restWaveCenters, alphaModelPixels, s=0)

    # Calculate residuals
    xgrid = numpy.outer(obsWaveCenters,1.0/(restWaveCenters))
    zevo = numpy.exp(-alphaModelPixels*numpy.power(xgrid,beta))
    model = numpy.outer(obsModel(obsWaveCenters),restModel(restWaveCenters))*zevo
    res = numpy.zeros(shape=flux.shape)
    res[iGood] = model[iGood] - flux[iGood]

    # Save HDF5 file with results
    grp = outfile.create_group('T')
    dset = grp.create_dataset('y', data=obsModelPixels)
    dset = grp.create_dataset('x', data=obsWaveCenters)

    grp = outfile.create_group('C')
    dset = grp.create_dataset('y', data=restModelPixels)
    dset = grp.create_dataset('x', data=restWaveCenters)

    # Save 1D plots
    fig = plt.figure(figsize=(10,6))

    #plt.plot(restWaveCenters, restModelPixels, '+')
    for iz,z in enumerate([0,1,2,3]): 
        ls = '-' #if iz % 2 == 0 else '.'
        x = (1+z)
        plt.plot(restWaveCenters, restModelPixels*numpy.exp(-alphaModelPixels*(x**beta)),ls=ls)
    plt.xlim([restWaveCenters[0],restWaveCenters[-1]])
    plt.xlabel(r'Rest Wavelength $\lambda_{rest}$')
    plt.ylabel(r'$C(\lambda_{rest},z=%.1f)$' % z)

    fig.savefig(args.output+'-cont.png', bbox_inches='tight')

    fig = plt.figure(figsize=(10,6))

    plt.plot(restWaveCenters, alphaModelPixels, '+')
    plt.xlim([restWaveCenters[0],restWaveCenters[-1]])
    plt.xlabel(r'Rest Wavelength $\lambda_{rest}$')
    plt.ylabel(r'$\alpha(\lambda_{rest})$')

    fig.savefig(args.output+'-alpha.png', bbox_inches='tight')

    fig = plt.figure(figsize=(20,6))

    plt.axhline(y=1, xmin=0, xmax=1, color='gray', linestyle='--')
    plt.plot(obsWaveCenters, obsModelPixels, '+')
    plt.xlim([obsWaveCenters[0],obsWaveCenters[-1]])
    plt.xlabel(r'Observed Wavelength $\lambda_{obs}$')
    plt.ylabel(r'$T(\lambda_{obs})$')
    plt.ylim([.5, 1.5])

    fig.savefig(args.output+'-trans.png', bbox_inches='tight')

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

    outfile.close()

if __name__ == '__main__':
    main()