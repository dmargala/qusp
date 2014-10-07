#!/usr/bin/env python
import argparse
import subprocess
import qusp

def build_image_url(target, width=256, height=256, opt='IOG', scale=.1):
    """
    Contructs a image cutout url for this target.

    see http://skyserver.sdss3.org/dr10/en/help/docs/api.aspx#cutout
    """
    base_url = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx'

    options = {}
    options['ra'] = target['ra']
    options['dec'] = target['dec']
    options['width'] = width
    options['height'] = height
    options['scale'] = scale
    options['opt'] = opt

    api_options = '&'.join(
        ['%s=%s' % (key, value) for key, value in options.iteritems()])
    full_url = base_url + '?' + api_options

    return full_url

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="target list")
    parser.add_argument("--ra-col", type=int, default=1,
                        help="ra column")
    parser.add_argument("--dec-col", type=int, default=2,
                        help="dec column")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't execute commands")
    parser.add_argument("--out-prefix", type=str, default='',
                        help="output prefix")
    args = parser.parse_args()

    fields = [('ra', float, args.ra_col), ('dec', float, args.dec_col)]
    targets = qusp.target.load_target_list(args.input, fields)

    for target in targets:
        url = build_image_url(target)
        cmd = ['curl', url, '-o', args.out_prefix+'%s.jpg'%target['target']]
        if args.dry:
            print ' '.join(cmd)
        else:
            subprocess.call(cmd)

if __name__ == '__main__':
    main()
