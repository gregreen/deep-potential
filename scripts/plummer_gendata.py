#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import json


def sample_df(n_samples, max_dist=None):
    """
    Returns phase-space locations sampled from the Plummer sphere
    distribution function. The shape of the output is
    (n_samples, 6).
    """
    # Instantiate Plummer sphere class
    plummer_sphere = toy_systems.PlummerSphere()
    x,v = plummer_sphere.sample_df(n_samples)
    if max_dist is not None:
        r2 = np.sum(x**2, axis=1)
        idx = (r2 < max_dist**2)
        x = x[idx]
        v = v[idx]
    return np.concat([x.astype('f4'), v.astype('f4')], axis=1)


def save_data(data, fname):
    o = {'eta': data.numpy().tolist()}
    with open(fname, 'w') as f:
        json.dump(o, f)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Deep Potential: Generate Plummer-sphere data.'
        add_help=True
    )
    parser.add_argument('-n', type=int, required=True, help='# of points.')
    parser.add_argument('-o', type=str, required=True, help='Output filename.')
    parser.add_argument('-d', type=float, default=10., help='Max. distance.')
    args = parser.parse_args()

    n_tot = 0
    data = []
    while n_tot < args.n:
        data.append(sample_df(int(1.2 * args.n), max_dist=args.d))
        n = data[-1].shape[0]
        n_tot += n
        print(f'Drew {n} samples.')
    data = np.concat(data, axis=0)[:args.n]
    
    print('Saving data.')
    save_data(data, args.o)

    return 0

if __name__ == '__main__':
    main()

