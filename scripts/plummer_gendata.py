#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import h5py

import toy_systems


def df_ideal(eta):
    q,p = np.split(eta, 2, axis=1)

    r2 = np.sum(q**2, axis=1)
    v2 = np.sum(p**2, axis=1)

    Phi = -(1+r2)**(-1/2)
    E = v2/2 + Phi

    f = np.clip(-E, 0, np.inf)**(7/2)

    A = 24 * np.sqrt(2.) / (7. * np.pi**3)

    return A * f


def calc_ideal_loss(eta):
    return -np.mean(np.log(df_ideal(eta)))


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
    return np.concatenate([x.astype('f4'), v.astype('f4')], axis=1)


def save_data(data, fname):
    with h5py.File(fname, 'w') as f:
        f.create_dataset('eta', data=data, compression='lzf', chunks=True)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Deep Potential: Generate Plummer-sphere data.',
        add_help=True
    )
    parser.add_argument(
        '-n',
        type=int,
        required=True,
        help='# of points.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output filename.'
    )
    parser.add_argument(
        '-d', '--max-dist',
        type=float,
        default=10.,
        help='Max. distance.'
    )
    args = parser.parse_args()

    n_tot = 0
    data = []
    while n_tot < args.n:
        data.append(sample_df(int(1.2 * args.n), max_dist=args.max_dist))
        n = data[-1].shape[0]
        n_tot += n
        print(f'Drew {n} samples within maximum distance.')
    data = np.concatenate(data, axis=0)[:args.n]

    loss = calc_ideal_loss(data)
    print(f'Ideal loss: {loss:.6f}')

    print('Saving data.')
    save_data(data, args.output)

    return 0

if __name__ == '__main__':
    main()

