#!/usr/bin/env python

from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import progressbar


def get_training_progressbar_fn(n_steps, loss_history, opt):
    widgets = [
        progressbar.Bar(),
        progressbar.Percentage(), ' |',
        progressbar.Timer(), '|',
        progressbar.AdaptiveETA(), '|',
        progressbar.Variable('loss', width=6, precision=4), ', ',
        progressbar.Variable('lr', width=8, precision=3)
    ]
    bar = progressbar.ProgressBar(max_value=n_steps, widgets=widgets)

    def update_progressbar(i):
        loss = np.mean(loss_history[-50:])
        lr = float(opt._decayed_lr(tf.float32))
        bar.update(i+1, loss=loss, lr=lr)

    return update_progressbar


def batch_function(f, batch_size, base_library=tf):
    def g(x, *args, **kwargs):
        o = []
        n_data = x.shape[0]
        for batch in base_library.split(x, range(0,n_data,batch_size)):
            o.append(f(batch, *args, **kwargs))
        #for k in range(0, n_data, batch_size):
        #    b0,b1 = k, k+batch_size
        #    o.append(f(x[b0:b1], *args, **kwargs))
        return base_library.concatenate(o)


def main():
    return 0

if __name__ == '__main__':
    main()

