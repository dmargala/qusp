#!/usr/bin/env python
import argparse
import subprocess
import qusp

def targetImageURL(target, w=256, h=256, opt='IOG', scale=.1):
    # see http://skyserver.sdss3.org/dr10/en/help/docs/api.aspx#cutout
    baseURL = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx'

    options = {}
    options['ra'] = target.ra
    options['dec'] = target.dec
    options['width'] = w
    options['height'] = h
    options['scale'] = scale
    options['opt'] = opt

    optionString = '&'.join(['%s=%s' % (key,value) for key,value in options.iteritems()])
    combinedUrl = baseURL + '?' + optionString

    return combinedUrl

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--input", type=str, default=None,
        help="target list")
    args = parser.parse_args()

    targets = qusp.readTargetList(args.input,fields=[('ra',float),('dec',float)])

    for target in targets:
        url = targetImageURL(target)
        cmd = ['curl',url,'-o','%s.jpg'%str(target)]
        subprocess.call(cmd)

if __name__ == '__main__':
    main()