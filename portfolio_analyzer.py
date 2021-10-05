#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:54:45 2021

@author: federico
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import urllib.request
import zipfile

def compute_returns_volat_sharpe(portfolio, risk_free_ret=0):
    returns = portfolio.val_tot_hist.pct_change()
    returns = returns.dropna()              # Discrete returns
    log_returns = np.log(1. + returns)      # Logarithmic returns
    if isinstance(risk_free_ret, pd.Series):
        rf_ret = risk_free_ret.loc[returns.index]
    else:
        rf_ret = risk_free_ret
    exc_ret = returns - rf_ret              # Excess returns
    exc_ret_mean = exc_ret.mean()           # Mean excess return
    volat = np.sqrt(252) * exc_ret.std()    # Volatility
    sharpe = 252. * exc_ret_mean / volat    # Sharpe ratio
    return returns, log_returns, exc_ret, exc_ret_mean, volat, sharpe

def compute_sortino(portfolio, min_acc_ret):  # Sortino ratio
    exc_ret_mean = 252. * portfolio.exc_ret_mean
    exc_ret_std_dnsd = np.sqrt(252) * portfolio.exc_ret[portfolio.exc_ret < 0].std()
    return exc_ret_mean / exc_ret_std_dnsd

def compute_star(portfolio, level):  # STAR ratio
    exc_ret_mean = portfolio.exc_ret.mean()
    value_at_risk = portfolio.exc_ret.quantile(level)
    exp_tail_loss = portfolio.exc_ret[portfolio.exc_ret < value_at_risk].mean()
    return - exc_ret_mean / exp_tail_loss

def compute_turnover(portfolio):  # turnover
    pf_val_stk_fill = portfolio.val_stk_hist.fillna(0)
    pf_weights = pf_val_stk_fill.divide(portfolio.val_tot_hist.iloc[1:], axis=0)
    pf_weights_diff = pf_weights.diff()
    pf_weights_diff = pf_weights_diff.dropna()
    turnover = pf_weights_diff.abs().sum(axis=1)
    return turnover

def compute_alpha_beta(portfolio, mkt_ret_df, risk_free_ret=0):  # alpha and beta factors
    if isinstance(risk_free_ret, pd.Series):
        rf_ret = risk_free_ret.loc[portfolio.returns.index]
    else:
        rf_ret = risk_free_ret
    mkt_ret_df_win = mkt_ret_df.loc[portfolio.returns.index]
    y = np.array(portfolio.returns.subtract(rf_ret, axis=0).values, dtype=float).reshape(-1, 1)
    x = np.array(mkt_ret_df_win.subtract(rf_ret, axis=0).values, dtype=float).reshape(-1, 1)
    x = sm.add_constant(x, prepend=True)
    ols = sm.OLS(y, x)
    ols_result = ols.fit()
    alpha = 252. * ols_result.params[0]
    beta = ols_result.params[1]
    pvalues = ols_result.pvalues
    return alpha, beta, pvalues

def compute_drawdown(portfolio):  # drawdown
    previous_peak = portfolio.val_tot_hist.cummax()
    drawdown = (portfolio.val_tot_hist - previous_peak) / previous_peak
    return drawdown

def download_ff_coeffs(n_factors):
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/North_America_3_Factors_Daily_CSV.zip'
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    first_file = zip_file_object.namelist()[0]
    file = zip_file_object.open(first_file)
    content = file.read()
    
    
    
    
    