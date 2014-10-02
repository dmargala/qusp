#!/usr/bin/env python
import argparse

import numpy
import h5py
import scipy.interpolate

def main():

    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--input", type=str, default=None,
        help="stacked spectrum file")
    parser.add_argument("--verbose", action="store_true",
        help="print verbose output")
    parser.add_argument("-n", type=int, default=1000,
        help="max number of iterations for fit")
    parser.add_argument("-o","--output", type=str, default=None,
        help="output file name")
    args = parser.parse_args()

    infile = h5py.File(args.input,'r')

    hists2d = infile['hists2d']
    zmin = hists2d.attrs['zmin']
    zmax = hists2d.attrs['zmax']

    wfluxmean = hists2d['wfluxmean'].value
    wfluxvar = hists2d['wfluxvar'].value

    npixels = wfluxmean.shape[0]
    nzbins = wfluxmean.shape[1]

    dz = float(zmax - zmin)/(nzbins)
    # x,y grid characterizing pixel bin centers
    x = 3500.26*numpy.power(10, 1e-4*(numpy.arange(0, npixels)))
    y = numpy.linspace(zmin+0.5*dz, zmax-0.5*dz, nzbins, endpoint=True)
    X,Y = numpy.meshgrid(x,y, indexing='ij')

    class QuasarContinuumModel:
        def __init__(self, wavemin=500, wavemax=7000, npixels=20):
            self.wavemin=wavemin
            self.wavemax=wavemax
            self.npixels=npixels
            # self.wave = numpy.logspace(numpy.log10(wavemin),numpy.log10(wavemax),npixels,endpoint=True)
            self.wave = numpy.concatenate((numpy.linspace(500,1300,4,endpoint=False),numpy.linspace(1300,3500,22,endpoint=False),numpy.linspace(3500,7000,4,endpoint=True)))
            self.npixels = len(self.wave)
            print self.wave
        def getPredication(self, params):
            flux = numpy.array(params, dtype=numpy.float32)
            restModel = scipy.interpolate.UnivariateSpline(self.wave, flux, w=None, k=3, s=0)
            def obsModel(wave, z):
                return restModel(wave/(1+z))
            return obsModel(X.flatten(),Y.flatten())

    model = QuasarContinuumModel()

    data = wfluxmean.flatten()
    variance = wfluxvar.flatten()
    sigma = numpy.sqrt(variance)
    nonzero = sigma > 0

    # define our chisquare function
    def chiSquare(*params):
        print params
        predicated = model.getPredication(params)
        residuals = (predicated[nonzero] - data[nonzero])/sigma[nonzero]
        chisq = numpy.dot(residuals,residuals)
        return chisq

    # initialize Minuit
    from iminuit import Minuit
    print_level = 1 if args.verbose else 0
    fitParameters = [ ]
    fitOpts = { }
    for i in range(model.npixels):
        name = 'p%d'%i
        fitParameters.append(name)
        fitOpts[name] = 1
        fitOpts['error_'+name] = 1
    # we fit to chisq here (not -logL) so errordef = 1.0
    engine = Minuit(chiSquare,forced_parameters=fitParameters,errordef=1.0,print_level=print_level,**fitOpts)
    # do the minimization
    minimum = engine.migrad(ncall=args.n)
    if not minimum[0]['has_valid_parameters']:
        raise RuntimeError('MINUIT migrad failed.')

    # print results    
    print 'parameters', engine.parameters
    print 'args', engine.args
    print 'value', engine.values

    print 'minimum', minimum

    if args.output:
        outfile = h5py.File(args.output,'w')
        bestfitModel = model.getPredication(engine.args).reshape(npixels,nzbins)
        outfile.create_dataset('fitresults', data=bestfitModel)
        outfile.close()

        import plotstack

        vmin=0
        vmax=numpy.percentile(bestfitModel,99)
        savename = 'fitresults'
        plotstack.plothist2d(bestfitModel, savename, zmin, zmax, vmin, vmax, label='bestfitModel')


if __name__ == '__main__':
    main()