#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import json


def load_data(fname):
    with open(fname, 'r') as f:
        o = json.load(f)
    d = np.array(o['eta'], dtype='f4')
    return d


def save_data(data, fname):
    o = {'eta': data.tolist()}
    with open(fname, 'w') as f:
        json.dump(o, f)


def spherical_cut(eta, origin, radius):
    r2 = np.sum((eta[:,:3]-origin[None,:])**2, axis=1)
    idx = r2 < radius**2
    return idx


def cylindrical_cut(eta, origin, R_max, z_max):
    R2 = np.sum((eta[:,:2]-origin[None,:2])**2, axis=1)
    dz = np.abs(eta[:,3] - origin[None,3])
    idx = (R2 < R_max**2) & (dz < z_max)
    return idx


def cartesian_cut(eta, origin, xyz_max):
    idx = (np.abs(eta[:,:3] - origin[None,:]) < xyz_max[None,:])
    return idx


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Deep Potential: Apply selection functions to data.',
        add_help=True
    )
    parser.add_argument(
        '--input',
        type=str, required=True,
        help='Input filename.'
    )
    parser.add_argument(
        '--output',
        type=str, required=True,
        help='Output filename.'
    )
    parser.add_argument(
        '--origin',
        type=float, nargs=3,
        required=True,
        help='Location of observer (x,y,z).'
    )
    parser.add_argument(
        '--spherical',
        type=float,
        help='Maximum distance from observer.'
    )
    parser.add_argument(
        '--cylindrical',
        type=float, nargs=2,
        help='Maximum cylindrical radius & vertical separation from observer.'
    )
    parser.add_argument(
        '--cartesian',
        type=float, nargs=3,
        help='Maximum separation in x, y & z from observer.'
    )
    args = parser.parse_args()

    origin = np.array(args.origin)

    data = load_data(args.input)
    print(f'Loaded {data.shape[0]} data points.')

    if args.spherical is not None:
        idx = spherical_cut(data, origin, args.spherical)
        data = data[idx]
        print(f'After spherical cut: {data.shape[0]} data points remain.')

    if args.cylindrical is not None:
        idx = cylindrical_cut(data, origin, *args.cylindrical)
        data = data[idx]
        print(f'After cylindrical cut: {data.shape[0]} data points remain.')

    if args.cartesian is not None:
        xyz_max = np.array(args.cartesian)
        idx = cartesian_cut(data, origin, xyz_max)
        data = data[idx]
        print(f'After cartesian cut: {data.shape[0]} data points remain.')

    print(f'Saving {data.shape[0]} data points ...')
    save_data(data, args.output)

    return 0

if __name__ == '__main__':
    main()
