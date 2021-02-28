# (c) 2021, Christoph Schmidt-Hieber
# See LICENSE

import os
import requests
import fitdecode
import re
import datetime
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.stats import pearsonr
from scipy.stats import linregress
from scipy.stats import norm

import matplotlib
import matplotlib.pyplot as plt

import argparse

def parse_filename(file):
    header = file.headers.get('content-disposition')
    if not header:
        return file.url.rsplit('/', 1)[1]
    fn = re.findall('filename=(.+)', header)
    return fn[0]


def get_fitfile(url, destdir="data"):
    if not os.path.exists(destdir):
        os.mkdir(destdir)
    file = requests.get(url, allow_redirects=True)
    filename = parse_filename(file)
    dstname = os.path.join(destdir, filename)
    if not os.path.exists(dstname):
        print("Writing to " + dstname)
        with open(dstname, 'wb') as fz:
            fz.write(file.content)
    return dstname


def dump_fitfile(filepath):
    dump = ""
    with fitdecode.FitReader(filepath, keep_raw_chunks=True) as fit:
        for frame in fit:
            # The yielded frame object is of one of the following types:
            # * fitdecode.FitHeader
            # * fitdecode.FitDefinitionMessage
            # * fitdecode.FitDataMessage
            # * fitdecode.FitCRC        # * fitdecode.FitHeader
            
            if isinstance(frame, fitdecode.FitDataMessage):
                dump += frame.name + '\n'
                for field in frame.fields:
                    dump += '\t' + field.name + ' ' + str(field.value) + '\n'
    return dump


def parse_fitfile(filepath, resample_dt=0.1, verbose=False):
    timeseries = []
    assert(resample_dt == 0.1)
    t0 = None
    with fitdecode.FitReader(filepath) as fit:
        for frame in fit:
            # The yielded frame object is of one of the following types:
            # * fitdecode.FitHeader
            # * fitdecode.FitDefinitionMessage
            # * fitdecode.FitDataMessage
            # * fitdecode.FitCRC        # * fitdecode.FitHeader
            if isinstance(frame, fitdecode.FitDataMessage):
                if frame.name == 'device_info':
                    for field in frame.fields:
                        if verbose and ( \
                            ('manufacturer' in field.name) or \
                            ('product' in field.name) or \
                            ('device_type' in field.name)):
                            print(field.name, field.value)
                if frame.name == 'record':
                    for nf, field in enumerate(frame.fields):
                        power = frame.get_value('power')
                        timestamp = frame.get_value('timestamp')
                        if power is None and verbose:
                            print("Power is None")
                        if t0 is None or timestamp != t0:
                            timeseries.append((timestamp, power))
                            t0 = timestamp
    df = pd.DataFrame(
        data=np.array(timeseries)[:,1].astype(float),
        index=np.array(timeseries)[:,0])
    df = df.resample('100ms').interpolate()
    return df, resample_dt


def align_fit_timeseries(fit1, fit2, dt):
    corr = correlate(fit1[0], fit2[0])
    lags = correlation_lags(fit1[0].shape[0], fit2[0].shape[0])
    best_lag = lags[corr.argmax()]
    if fit1.index[0] > fit2.index[0]:
        record_start_lag = np.where(fit2.index >= fit1.index[0])[0][0]
    else:
        record_start_lag = -np.where(fit1.index >= fit2.index[0])[0][0]
    net_lag = record_start_lag + best_lag
    return net_lag*dt, record_start_lag*dt, lags*dt


def xcorr(x, y, normed=True):
    correls = correlate(x, y)
    if normed:
        correls /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    return correls


def plot_unity_residuals(x, y, fig, ratio=4.0, distance=0.05, bins=None, ms=5):
    residuals = x - y
    imax = np.argmax(x+y)
    imin = np.argmin(x+y)
    xymax = (x[imax] + y[imax])/2.0
    xymin = (x[imin] + y[imin])/2.0
    xyamp = xymax-xymin
    xyoffset = xymax + distance*xyamp
    if bins is None:
        bins = int(np.ceil(np.log10(x.shape[0])))*10
    histo, bins = np.histogram(residuals, bins=bins, density=True)
    bins_rotate = bins/np.sqrt(2.0)
    scale = xyamp/(ratio*histo.max())
    linr = linregress(x, y)

    ax = fig.add_subplot(aspect='equal')
    ax.plot(x, y, 'o', ms=ms, mec='none', alpha=0.5)
    ax.plot(
        [xymin, xyoffset+scale*histo.max()],
        [xymin, xyoffset+scale*histo.max()], '--', color='k', alpha=0.5, label="Unity")
    ax.plot(
        [x.min(), x.max()],
        [linr.slope*x.min()+linr.intercept, linr.slope*x.max()+linr.intercept],
        '--', color='r', alpha=0.8, label="Linear regression"
    )
    leg = ax.legend(frameon=False, loc='upper left')
    base_trans = ax.transData
    tr = matplotlib.transforms.Affine2D().rotate_deg(-45).translate(
        xyoffset, xyoffset) + base_trans
    ax.bar(
        (bins_rotate[:-1]+bins_rotate[1:])/2.0,
        histo*scale, width=(bins_rotate[1]-bins_rotate[0]), ec='none', color='k', alpha=0.8,
        transform=tr)
    return ax, histo, bins


def plot_residual_distribution(x, y):
    residuals = x - y
    xfit = np.arange(
        residuals.min(), residuals.max(),
        (residuals.max()-residuals.min())/1000.0)
    loc, sigma = norm.fit(residuals)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(residuals, density=True, bins=200)
    ax.plot(xfit, norm.pdf(xfit, loc, sigma), '-r', alpha=0.8, lw=3)
    ax.axvline(0, ls='--', color='k', alpha=0.5)
    ax.axvline(loc, ls='--', color='r', alpha=0.5)
    prettify_plot(ax)
    ax.set_xlabel(r"$\Delta$ Power (W)")
    ax.set_ylabel(r"$P$")


def summarize_stats(x, y, sources):
    residuals = x-y
    print("Mean power ({0}): {1:.2f}W".format(sources[0], x.mean()))
    print("Mean power ({0}): {1:.2f}W".format(sources[1], y.mean()))
    print("Mean residual (\"offset\"): {0:.2f}W".format(residuals.mean()))
    print("Mean absolute residual (\"precision\"): {0:.2f}W".format(
        np.abs(residuals).mean()))
    print("Mean absolute residual after accounting for offset(\"precision\"): {0:.2f}W".format(
        np.abs(residuals-residuals.mean()).mean()))
    print("Mean relative residual after accounting for offset (\"precision\"): {0:.2f}%".format(
        100.0*np.abs(residuals-residuals.mean()).mean()/((x.mean()+y.mean())/2.0)))
    print("Residual variance: {0:.2f}W^2".format(np.var(residuals)))

def prettify_plot(ax):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fitfile", help="path to fit file", type=str)
    args = parser.parse_args()
    if not os.path.exists(args.fitfile):
        print("Could not find " + args.fitfile)
        sys.exit(1)
    print(dump_fitfile(args.fitfile))
