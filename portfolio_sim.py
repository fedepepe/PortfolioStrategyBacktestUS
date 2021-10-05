#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 12:23:10 2021

@author: federico
"""

import datetime
import pandas as pd
from stock_picker import StockPicker
from portfolio import Portfolio
from pathlib import Path
import csv
import time


def calculate_time(func):
    def wrapper(self, *args, **kwargs):
        start = time.time()
        # Run simulation
        func(self, *args, **kwargs)
        end = time.time()

        elapsed = end - start
        print('\n --- Simulation time: ' + time.strftime('%Mm %Ss', time.gmtime(elapsed)) + ' --- ')

    return wrapper


class PortfolioSimulator:

    def __init__(self, n_stk, n_obs, n_reb, algo, wght_mtd,
                 last_price_df, return_df, volat_df, mktcap_df=None, bema_ret_df=None,
                 endow=1e6, idx_start=None, lag=1, nsel=0.15, trsctn_fee_fix=0, trsctn_fee_prop=10e-4,
                 risk_avers_factor=None, max_leverage=2., multi_proc=True, cv_opt_bw=False,
                 results_dir=None, results_tag='', results_date=None,
                 overwrite_results=True, save_stk_hist=True):
        self.n_stk = n_stk
        self.n_obs = n_obs
        self.n_reb = n_reb
        self.algo = algo
        self.wght_mtd = wght_mtd
        if type(self.wght_mtd) == str:  # Convert string to list
            self.wght_mtd = [self.wght_mtd]
        self.last_price_df = last_price_df
        self.return_df = return_df
        self.volat_df = volat_df
        self.mktcap_df = mktcap_df
        self.bema_ret_df = bema_ret_df
        if bema_ret_df is None:
            exc_ret_df = return_df
        else:
            exc_ret_df = return_df.subtract(bema_ret_df, axis=0)
        # TO DO: include also the volatility of the benchmark in the Sharpe ratio formula
        self.sharpe_df = exc_ret_df / volat_df
        self.endow = endow
        if idx_start is None:  # when n_obs is swept, idx_start can be set equal to
            self.idx_start = n_obs + lag - 1  # the maximum n. of observations, to have all
        else:  # simulation starting at the same trading day
            self.idx_start = max(idx_start, n_obs) + lag - 1
        self.lag = lag  # Time lag (in days) between last observation and of rebalancing
        self.nsel = nsel
        self.trsctn_fee_fix = trsctn_fee_fix
        self.trsctn_fee_prop = trsctn_fee_prop
        self.risk_avers_factor = risk_avers_factor
        self.max_leverage = max_leverage
        self.multi_proc = multi_proc
        self.cv_opt_bw = cv_opt_bw
        self.save_stk_hist = save_stk_hist

        # Initialize portfolio(s)
        self.init_portofolios(self.last_price_df.index[0])

        if results_dir is not None:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            self.results_dir = results_dir
            self.results_filenames = {}
            if results_date is None:
                timestamp_string = datetime.date.today().strftime('%Y-%m-%d')
            else:
                timestamp_string = results_date
            for wm in self.wght_mtd:
                self.results_filenames[wm] = results_dir + results_tag + '_' + \
                                             timestamp_string + '_' + self.algo + '_' + wm + '.txt'
                if overwrite_results:
                    file_handle = open(self.results_filenames[wm], 'w')
                    file_handle.close()

    def init_portofolios(self, timestamp):
        # Create portfolio(s)
        self.pf_dict = {}
        for wm in self.wght_mtd:
            self.pf_dict[wm] = Portfolio(self.endow)
            # Record the initial cash in the portfolio history
            self.pf_dict[wm].update_hist(timestamp)

    def allocate(self, timestamp=None, price_curr_dct=None):
        datetimeidx = self.return_df.index

        # If no argument is passed, simply compute allocation at the end of 
        # the given time frame of observations
        if timestamp is None:
            timestamp = datetimeidx[-1]
        if price_curr_dct is None:
            price_curr_dct = self.last_price_df.loc[timestamp].to_dict()

        # Exclude a number 'lag' of samples before and including the current timestamp
        # and then get the last n_obs observations
        ret_df_win = self.return_df[datetimeidx <= timestamp]
        ret_df_win = ret_df_win[:-self.lag].tail(self.n_obs)
        sharpe_df_win = self.sharpe_df[datetimeidx <= timestamp]
        sharpe_df_win = sharpe_df_win[:-self.lag].tail(self.n_obs)

        if self.mktcap_df is None:
            cap_df_win = None
        else:
            cap_df_win = self.mktcap_df[datetimeidx <= timestamp]
            cap_df_win = cap_df_win[:-self.lag].tail(self.n_obs)

        if self.bema_ret_df is None:
            bema_df_hist = None
        else:
            bema_df_hist = self.bema_ret_df[self.bema_ret_df.index <= timestamp][:-self.lag]

        # Create stock picker object
        stock_picker = StockPicker(self.n_stk, self.algo, ret_df_win, sharpe_df_win,
                                   cap_df_win, self.nsel)

        # Perform stock selection
        stock_sel_df = stock_picker.pick_stocks(self.multi_proc, self.cv_opt_bw)
        stock_sel = stock_sel_df.index

        # Add column with last price
        stock_sel_df['Price'] = [price_curr_dct[tckr] for tckr in stock_sel]

        # Use historical returns and volatilities up to the current rebalancing day 
        # for Markowitz' portfolio optimization
        ret_sel_df_hist = self.return_df.loc[datetimeidx <= timestamp, stock_sel][:-self.lag]
        vol_sel_df_hist = self.volat_df.loc[datetimeidx <= timestamp, stock_sel][:-self.lag]

        # Compute portfolio allocation
        pf_alloc_dct = {}
        for wm in self.pf_dict:
            pf_alloc_dct[wm] = self.pf_dict[wm].compute_weights(stock_sel_df, wm,
                                                                self.risk_avers_factor,
                                                                self.max_leverage,
                                                                self.mktcap_df.loc[timestamp],
                                                                ret_sel_df_hist, vol_sel_df_hist,
                                                                bema_df_hist,
                                                                self.trsctn_fee_fix,
                                                                self.trsctn_fee_prop)
        return pf_alloc_dct

    @calculate_time
    def backtest(self, start_date=None, stop_date=None):
        # Make a copy of last price dataframe that can be manipulated
        last_price_df = self.last_price_df.copy()

        # Select only data samples between start and stop date, if given
        if start_date is not None:
            last_price_df = last_price_df.loc[last_price_df.index >= start_date]
        if stop_date is not None:
            last_price_df = last_price_df.loc[last_price_df.index <= stop_date]

        # Check if there are enough samples for the first iteration
        n_samples = last_price_df.shape[0]
        if n_samples <= self.n_obs:
            return None

        # Get full list of days to be simulated and those when rebalancing occurs
        timestamps = last_price_df.index[self.idx_start:]
        timestamps_reb = last_price_df.index[self.idx_start::self.n_reb]

        print(f" --- #obs: {self.n_obs}, #reb: {self.n_reb}, #stk: {self.n_stk}. "
              f"Total time steps: {timestamps.size}. --- ")

        # Build dictionary with last price for computing portfolio value
        # To this purpose, fill NaNs with last valid data, in order to
        # get an approximated value of the wealth even if not all asset prices
        # are available
        last_price_df_fill = last_price_df.fillna(method='ffill')

        # (Re-)initialize portfolio(s)
        self.init_portofolios(last_price_df.index[self.idx_start - 1])

        step = 1
        for timestamp in timestamps:

            price_curr_dct_fill = last_price_df_fill.loc[timestamp].to_dict()

            for wm in self.pf_dict:
                self.pf_dict[wm].update_value(price_curr_dct_fill)

            if timestamp in timestamps_reb:
                # For the transaction, we use the last available prices without
                # filling NaNs, thus excluding selected stocks for which we
                # don't have a valid price
                price_curr_dct = last_price_df.loc[timestamp].to_dict()

                # Compute new portfolio allocation(s)
                pf_alloc_dct = self.allocate(timestamp, price_curr_dct)

                # Execute trading
                for wm in self.pf_dict:
                    self.pf_dict[wm].rebalance(pf_alloc_dct[wm], price_curr_dct,
                                               self.trsctn_fee_fix, self.trsctn_fee_prop)

            for wm in self.pf_dict:
                self.pf_dict[wm].update_hist(timestamp, self.save_stk_hist)

            if step == 5:
                print(' Done step', end=' ')

            if step % 5 == 0:
                print(f'{step}', end='..')

            step += 1

    def analyze(self, mkt_ret_df, risk_free_ret=0):
        for wm in self.pf_dict:
            self.pf_dict[wm].analyze(mkt_ret_df, risk_free_ret)

    def print_results(self):
        for wm in self.pf_dict:
            if self.pf_dict[wm].returns is None:
                print('\n --- Error! Run portfolio analysis first! --- ')
            else:
                filename = self.results_filenames[wm]

                pf = self.pf_dict[wm]

                file_handle = open(filename, 'a')
                data_str = (
                    f'{self.n_stk}\t{self.n_obs}\t{self.n_reb}\t{self.algo}\t{wm}\t{100 * pf.alpha:.4f}\t'
                    f'{pf.pvalues[0]:.4f}\t{pf.beta:.4f}\t{100 * pf.volat:.4f}\t{pf.sharpe:.4f}\t{pf.sortino:.4f}\t'
                    f'{pf.star:.6f}\t{-100 * min(pf.drawdown):.4f}\t{100 * pf.turnover.mean():.4f}\n')
                file_handle.write(data_str)
                file_handle.close()

    def load_results(self, wght_mtd=None, filename=None):
        if filename is None:
            filename = self.results_filenames[wght_mtd]
        else:
            filename = self.results_dir + filename

        with open(filename, newline='') as results_handle:
            results_reader = csv.reader(results_handle, delimiter='\t')
            col_names = ['Nstk', 'Nobs', 'Nreb', 'algo', 'wght', 'alpha', 'pval',
                         'beta', 'volat', 'sharpe', 'sortino', 'star', 'maxdd', 'to']
            results_df = pd.DataFrame(data=results_reader, columns=col_names)

        # Convert some columns to numerics
        col_num = ['Nstk', 'Nobs', 'Nreb', 'alpha', 'pval', 'beta', 'volat', 'sharpe',
                   'sortino', 'star', 'maxdd']
        results_df[col_num] = results_df[col_num].apply(pd.to_numeric)

        return results_df
