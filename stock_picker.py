#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import pandas as pd
import numpy as np
import multiprocessing
import stock_picker_tools as pkr_tools
import compute_df
import compute_hsr_index

class StockPicker:

    def __init__(self, n_stck, algo, ret_df, sharpe_df, mktcap_df=None, nsel=0.15):
        self.n_stck = n_stck
        self.algo = algo
        self.ret_df = ret_df
        self.sharpe_df = sharpe_df
        self.cap_df = mktcap_df
        self.nsel = nsel

    def use_momentum(self, improved=True):
        if improved:
            mom_df = self.sharpe_df
        else:
            mom_df = self.ret_df
        mom_df = mom_df.mean(skipna=False)
        mom_df = mom_df.sort_values(ascending=False)
        mom_df = mom_df.head(self.n_stck)
        mom_df = mom_df.to_frame()
        mom_df.columns = ['metric']
        return mom_df

    def build_high_sr_index(self, improved):
        # Create artificial (high Sharpe ratio) index
        hsr_idx = pd.Series()
        for d in self.sharpe_df.index:
            hsr_idx_curr = compute_hsr_index.compute_hsr_index(self, d, improved)
            hsr_idx_curr = pd.Series(data=hsr_idx_curr, index=[d])
            hsr_idx = pd.concat([hsr_idx, hsr_idx_curr])

        return hsr_idx

    def use_sev(self, improved=True, multi_proc=True, cv_opt_bw=False):
        # Build artificial high Sharpe ratio index
        hsr_idx = self.build_high_sr_index(improved)

        # Compute dataframe to be used
        df_to_use = compute_df.compute_df(self, hsr_idx, improved)

        tickers = self.ret_df.columns

        if multi_proc:
            # Reset shared dictionary of SEV values
            shrd_dct = multiprocessing.Manager().dict()
            shrd_dct.clear()

            jobs = []
            jobs_running = 0
            max_jobs_running = multiprocessing.cpu_count()

            # Compute SEV metric for each stock using multiprocessing
            for tkr_cnk in pkr_tools.chunks(tickers, int(len(tickers) / max_jobs_running)):
                args = (shrd_dct, df_to_use[tkr_cnk], hsr_idx, cv_opt_bw)
                p = multiprocessing.Process(target=pkr_tools.compute_sev_multiproc,
                                            args=args)
                jobs.append(p)
                p.start()

                jobs_running += 1
                if jobs_running >= max_jobs_running:
                    while jobs_running >= max_jobs_running:
                        jobs_running = 0
                        for p in jobs:
                            jobs_running += p.is_alive()

            for p in jobs:
                p.join()

            # Convert dict to df, remove NaNs, sort by descending SEV value and pick top entries
            sev_df = pd.DataFrame(shrd_dct.values(), index=shrd_dct.keys(), columns=['metric'])
        else:
            sev_df = pd.DataFrame(np.nan, index=tickers, columns=['metric'])
            for tkr in tickers:
                sev_df.loc[tkr, 'metric'] = pkr_tools.compute_sev(x=df_to_use[tkr].values,
                                                                  y=hsr_idx.values,
                                                                  cv_opt_bw=cv_opt_bw)

        sev_df = sev_df.dropna()
        sev_df = sev_df.sort_values(ascending=False, by='metric')
        return sev_df.head(self.n_stck)

    def pick_stocks(self, multi_proc=True, cv_opt_bw=False):
        if self.algo == 'mom' or self.algo == 'mtm':
            stock_df = self.use_momentum(improved=False)
        elif self.algo == 'rmmom' or self.algo == 'rmmtm':
            stock_df = self.use_momentum()
        elif self.algo == 'sev':
            stock_df = self.use_sev(improved=False, multi_proc=multi_proc, cv_opt_bw=cv_opt_bw)
        elif self.algo == 'sev+':
            stock_df = self.use_sev(multi_proc=True, cv_opt_bw=False)
        else:
            stock_df = None

        return stock_df
