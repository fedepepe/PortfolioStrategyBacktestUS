#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:42:40 2021

@author: federico
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd


# Plot cumulative portfolio wealth
def plot_cum_wealth(portfolio, figure=None, descr=''):
    x_plot = portfolio.val_tot_hist.index
    y_plot = portfolio.val_tot_hist.values

    rnd_col = (np.random.rand(), np.random.rand(), np.random.rand())

    if descr == '':
        descr = 'portfolio strategy'
    else:
        descr = 'portfolio strategy (' + descr + ')'

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)

        ax.plot(x_plot, y_plot, '-s', color=rnd_col, label=descr, markersize=5)
        ax.set_xticks(portfolio.val_tot_hist.index)
        ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in x_plot],
                           rotation=45, ha='right')
        ax.set_ylabel('Cumulative wealth', fontsize=12)
    else:
        ax = figure.axes[0]
        ax.plot(x_plot, y_plot, '-s', color=rnd_col, label=descr, markersize=5)

    return figure


# Plot market index
def plot_mkt_index(portfolio, mkt_idx_df, figure):
    endow = portfolio.val_tot_hist[0]
    mkt_idx_df_plot = mkt_idx_df.loc[portfolio.val_tot_hist.index[0]:]
    figure.axes[0].plot(mkt_idx_df_plot.index,
                        mkt_idx_df_plot / mkt_idx_df_plot.iloc[0] * endow,
                        '--b', label='Market index')

    return figure


# Plot stock closing prices
def plot_stk_price(last_price_df, tckr):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(last_price_df.index, last_price_df[tckr], '-s')
    ax.set_xticks(last_price_df.index)
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in last_price_df.index],
                       rotation=45, ha='right')
    ax.set_ylabel('Closing price', fontsize=12)
    figure.suptitle(tckr, fontsize=16)


# Plot cumulative portfolio wealth versus number of stocks
def plot_vs_nstock(results_df, label, figure=None, descr=''):
    x_plot = results_df['Nstk'].values
    y_plot = results_df[label].values

    rnd_col = (np.random.rand(), np.random.rand(), np.random.rand())

    if descr == '':
        descr = 'portfolio strategy'
    else:
        descr = 'portfolio strategy (' + descr + ')'

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)

        ax.plot(x_plot, y_plot, '-s', color=rnd_col, label=descr, markersize=5)
        ax.set_xticks(results_df['Nstk'])
        ax.set_ylabel(label, fontsize=12)
    else:
        ax = figure.axes[0]
        ax.plot(x_plot, y_plot, '-s', color=rnd_col, label=descr, markersize=5)

    return figure


def make_grid(results_df, label):
    # Build dataframe with results loaded from file
    n_obs_ar = np.sort(np.unique(results_df['Nobs'].astype(int)))  # Number of past observations
    n_reb_ar = np.sort(np.unique(results_df['Nreb'].astype(int)))  # Rebalancing interval

    df = pd.DataFrame(np.nan, index=n_reb_ar, columns=n_obs_ar)

    for index, row in results_df.iterrows():
        df.loc[int(row['Nreb']), int(row['Nobs'])] = results_df.loc[index, label]
    return df.astype(float)


def build_watermelon_colormap(df, options):
    if options['val_max'] is not None:
        val_max = options['val_max']
    else:
        val_max = df.to_numpy().max()

    if options['val_min'] is not None:
        val_min = options['val_min']
    else:
        val_min = df.to_numpy().min()

    abs_max = max(abs(val_max), abs(val_min))

    val_max_norm = val_max / abs_max
    val_min_norm = val_min / abs_max

    cmax = 0.5 * (1. + val_max_norm)
    cmin = 0.5 * (1. + val_min_norm)

    # Rescale cmax to cmin to fit the given center value, corresponding to white
    if options['val_wht'] is not None:
        cp = 0.5 * (1. + options['val_wht'] / abs_max)
        if cmax == 1:
            cmax = 1.
            cmin = 0.5 / (1 - cp) * cmin + (0.5 - cp) / (1 - cp)
        elif cmin == 0:
            cmax = 0.5 / cp * cmax
            cmin = 0.

    cp = (0.5 - cmin) / (1.0 - cmin) if cmax == 1 else 0.5 / cmax

    vrlo = min(0.4 * (2.0 + cmin), 2.0 * (1.0 - cmin))
    vrhi = min(2.0 * (1.0 - cmax), 0.4 * (2.0 + cmax))
    vglo = min(2.0 * cmin, 0.4 * (3.0 - cmin))
    vghi = min(0.4 * (3.0 - cmax), 2.0 * cmax)

    if 0 <= cp <= 1:
        # This dictionary defines the colormap with transition from red to green
        cdict = {'red': ((0.0, vrlo, vrlo),  # set to 0.8 so its not too bright at 0
                         (cp, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                         (1.0, vrhi, vrhi)),  # no red at 1

                 'green': ((0.0, vglo, vglo),  # no green at 0
                           (cp, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                           (1.0, vghi, vghi)),  # set to 0.8 so its not too bright at 1

                 'blue': ((0.0, vglo, vglo),  # no blue at 0
                          (cp, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                          (1.0, vrhi, vrhi))  # no blue at 1
                 }
    else:
        # This dictionary defines the colormap with red or green shades only
        cdict = {'red': ((0.0, vrlo, vrlo),  # set to 0.8 so its not too bright at 0
                         (1.0, vrhi, vrhi)),  # no red at 1

                 'green': ((0.0, vglo, vglo),  # no green at 0
                           (1.0, vghi, vghi)),  # set to 0.8 so its not too bright at 1

                 'blue': ((0.0, min(vrlo, vglo), min(vrlo, vglo)),  # no blue at 0
                          (1.0, min(vrhi, vghi), min(vrhi, vghi)))  # no blue at 1
                 }

    # Create the colormap using the dictionary
    watermelon_cm = colors.LinearSegmentedColormap('GnRd', cdict)
    return watermelon_cm, val_max, val_min


def plot_heatmap(results_df, label, figure=None, options=None, descr=''):
    """ Plot heatmap of portfolio performance metric versus n. of observations and rebalancing interval
    label   : parameter to be plotted
    val_wht : value corresponding to white color """

    if options is None:
        options = {'val_max': None, 'val_min': None, 'val_wht': None}

    # Rearrange data on a 2-D dataframe for easy plotting
    df = make_grid(results_df, label)

    # Define the axes range
    dx = 0.5 * (df.columns[1] - df.columns[0])
    dy = 0.5 * (df.index[1] - df.index[0])
    extent = [df.columns.min() - dx, df.columns.max() + dx, df.index.min() - dy, df.index.max() + dy]

    # Build watermelon colormap
    watermelon_cm, val_max, val_min = build_watermelon_colormap(df, options)

    if figure is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
    else:
        ax = figure.axes[0]

    hm = ax.imshow(df.iloc[::-1], cmap=watermelon_cm, vmin=val_min, vmax=val_max, extent=extent)
    ax.set_xlabel('Window length [days]', fontsize=12)
    ax.set_ylabel('Rebalancing interval [days]', fontsize=12)
    ax.set_xticks(df.columns)
    ax.set_yticks(df.index)
    if descr == '':
        title = label
    else:
        title = label + ' (' + descr + ')'
    figure.suptitle(title, fontsize=16)
    figure.tight_layout()
    figure.colorbar(hm, ax=ax)
    figure.show()
    return figure, df
