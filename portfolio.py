#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 00:03:44 2021

@author: federico
"""

import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import scipy.optimize as opt
import portfolio_analyzer as pf_analyz
import warnings


def quadratic_util_fun(w, mu, cov_mat, phi):
    # w  : vector of weights
    # mu : vector of mean returns
    # cov_mat : covariance matrix of returns
    # phi : risk aversion coefficient

    if phi is None:
        # Compute risk aversion factor yielding the tangency portfolio
        # Sigma_np = cov_mat.to_numpy()
        # invSigma_np = np.linalg.inv(Sigma_np)
        # invSigma = pd.DataFrame(data = invSigma_np, index = cov_mat.index, columns = cov_mat.columns)
        # series_of_ones = pd.Series(1, index=mu.index)
        # phi0 = series_of_ones.dot(invSigma).dot(mu)
        # return phi0/2 * w.dot(cov_mat).dot(w) - w.dot(mu)  # Tangency portfolio
        # Alternative single-line (equivalent) code:
        return - w.dot(mu) / math.sqrt(abs(w.dot(cov_mat).dot(w)))  # Tangency portfolio
        # abs just prevents the run from failing due to numerical errors (causing w*Sigma*w being < 0)
    elif math.isinf(phi):
        return w.dot(cov_mat).dot(w)  # Minimum variance portfolio
    else:
        return phi / 2. * w.dot(cov_mat).dot(w) - w.dot(mu)  # Optimized portfolio


def shortfall_fun(w, ret_df, bema_ret_df=None, shortfall=-0.01, level=0.95):
    exc_ret_df = ret_df.values.dot(w.transpose())
    if bema_ret_df is not None:
        exc_ret_df = exc_ret_df - bema_ret_df.values

    mu = np.nanmean(exc_ret_df)
    sigma = np.nanstd(exc_ret_df)

    return norm.ppf(level) * sigma - mu + shortfall  # This must be <= 0


def leverage(w):
    return abs(w).sum()


class Portfolio:

    def __init__(self, endow):
        self.holdings_curr = {'cash': endow}  # Current # of stocks hold
        self.holdings_hist = pd.DataFrame()  # Historical # of stocks hold
        self.val_stk_curr = {'cash': endow}  # Current value of stocks hold
        self.val_stk_hist = pd.DataFrame()  # Historical value of stocks hold
        self.val_tot_curr = endow  # Current overall portfolio value
        self.val_tot_hist = pd.Series()  # Historical overall portfolio value
        self.returns = None
        self.logreturns = None
        self.sharpe = None
        self.sortino = None
        self.star = None
        self.turnover = None
        self.alpha, self.beta, self.pvalues = None, None, None
        self.drawdown = None

    def trade_stock(self, tckr, price, quantity, trsctn_fee_fix, trsctn_fee_prop):
        # Compute the transaction value
        trsctn_value = quantity * price

        # Exchange cash with stock
        self.holdings_curr['cash'] = self.holdings_curr['cash'] - trsctn_value
        self.val_stk_curr['cash'] = self.val_stk_curr['cash'] - trsctn_value
        # Subtract transaction costs
        trsctn_cost = trsctn_fee_fix + trsctn_fee_prop * abs(trsctn_value)
        self.holdings_curr['cash'] = self.holdings_curr['cash'] - trsctn_cost
        self.val_stk_curr['cash'] = self.val_stk_curr['cash'] - trsctn_cost
        # Update the number of stocks hold
        if tckr in self.holdings_curr:
            self.holdings_curr[tckr] = self.holdings_curr[tckr] + quantity
            self.val_stk_curr[tckr] = self.val_stk_curr[tckr] + trsctn_value
        else:
            self.holdings_curr[tckr] = quantity
            self.val_stk_curr[tckr] = trsctn_value

        # If there no stocks left, remove the entry from the portfolio dictionary
        if self.holdings_curr[tckr] == 0:
            del self.holdings_curr[tckr]
            del self.val_stk_curr[tckr]

        # Update the current portfolio value (change is only due to transaction costs)
        self.val_tot_curr = self.val_tot_curr - trsctn_cost

    def rebalance(self, stock_df_new, price_dct, trsctn_fee_fix, trsctn_fee_prop):
        tckrs_sel = stock_df_new.index.tolist()

        # First, liquidate open positions that are no longer needed
        tckrs_to_sell = [tckr for tckr in list(self.holdings_curr.keys()) if tckr not in tckrs_sel]
        tckrs_to_sell.remove('cash')

        for tckr in tckrs_to_sell:
            quantity = -self.holdings_curr[tckr]
            if not math.isnan(price_dct[tckr]):
                self.trade_stock(tckr, price_dct[tckr], quantity, trsctn_fee_fix, trsctn_fee_prop)

        # Then, update quantity of stocks already present in the portfolio
        tckrs_to_update = [tckr for tckr in list(self.holdings_curr.keys()) if tckr in tckrs_sel]
        for tckr in tckrs_to_update:
            quantity = stock_df_new.loc[tckr, 'Quantity'] - self.holdings_curr[tckr]
            if not math.isnan(price_dct[tckr]):
                self.trade_stock(tckr, price_dct[tckr], quantity, trsctn_fee_fix, trsctn_fee_prop)

        # Lastly, buy new stocks
        tckrs_to_buy = [tckr for tckr in tckrs_sel if tckr not in list(self.holdings_curr.keys())]
        for tckr in tckrs_to_buy:
            quantity = stock_df_new.loc[tckr, 'Quantity']
            self.trade_stock(tckr, price_dct[tckr], quantity, trsctn_fee_fix, trsctn_fee_prop)

    def update_value(self, price_dct):
        total_value = .0
        for tckr in self.holdings_curr:
            if tckr in price_dct:
                if not math.isnan(price_dct[tckr]):
                    stk_value = self.holdings_curr[tckr] * price_dct[tckr]
                    self.val_stk_curr[tckr] = stk_value
                    total_value = total_value + stk_value
                else:
                    print('Missing price for ' + tckr + '!')
            elif tckr == 'cash':
                total_value = total_value + self.holdings_curr['cash']

        self.val_tot_curr = total_value

    def update_hist(self, timestamp, save_stk_hist=True):
        if save_stk_hist:
            holdings_curr_df = pd.DataFrame(data=[list(self.holdings_curr.values())],
                                            index=[timestamp],
                                            columns=list(self.holdings_curr.keys()))
            self.holdings_hist = self.holdings_hist.append(holdings_curr_df)

            val_stk_curr_df = pd.DataFrame(data=[list(self.val_stk_curr.values())],
                                           index=[timestamp],
                                           columns=list(self.val_stk_curr.keys()))
            self.val_stk_hist = self.val_stk_hist.append(val_stk_curr_df)

        self.val_tot_hist = self.val_tot_hist.append(pd.Series(self.val_tot_curr, index=[timestamp]))

    @staticmethod
    def get_opt_weights(stock_idx, ret_df, allow_short_sell, risk_avers, max_lvrg,
                        bema_ret_df=None):

        series_of_ones = pd.Series(1, index=stock_idx)

        # Imposing sum of weights being equal to 1
        linear_constraint = opt.LinearConstraint(series_of_ones, 1, 1)

        # Imposing the restriction on maximum leverage
        nonlinear_constraint = opt.NonlinearConstraint(leverage, 1, max_lvrg)

        # Imposing the no short-selling restriction
        if allow_short_sell:
            bounds = opt.Bounds(-np.inf * series_of_ones, np.inf * series_of_ones)
        else:
            bounds = opt.Bounds(0 * series_of_ones, np.inf * series_of_ones)

        # Initial starting point is the equally-weighted portfolio
        n_stck = len(stock_idx)
        w0 = 1. / n_stck * series_of_ones

        method = 'meanvar'

        try:
            if method == 'meanvar':
                # Compute vector of mean returns and covariance matrix
                mu = ret_df.mean().values  # vector of mean returns
                cov_mat = ret_df.cov().values  # covariance matrix
                fun = quadratic_util_fun
                args = (mu, cov_mat, risk_avers)
            elif method == 'shortfall':
                fun = shortfall_fun
                args = (ret_df, bema_ret_df)

            opt_result = opt.minimize(fun, w0, args=args, method='trust-constr',
                                      options={'verbose': False, 'gtol': 1e-3, 'xtol': 1e-3},
                                      constraints=(linear_constraint, nonlinear_constraint),
                                      bounds=bounds)
            if method == 'shortfall' and opt_result.fun > 0:
                message = ("\n --- No portfolio found satisfying shortfall constraint. "
                           f"Decrease shortfall by {100 * opt_result.fun:.2f}% --- ")
                print(message)

            return pd.Series(opt_result.x, index=stock_idx)
        except Exception:
            message = (" --- Mean-variance optimization failed! "
                       "Switching to equally-weighted portfolio --- ")
            print(message)
            # breakpoint()
            return w0

    def compute_weights(self, stock_sel_df, wght_mtd, risk_avers, max_lvrg,
                        mktcap_df=None, ret_df=None, vol_df=None, bema_ret_df=None,
                        trsctn_fee_fix=0, trsctn_fee_prop=0):

        stock_df = stock_sel_df.copy()

        # Add columns with market capitalization or volatility, if needed
        if wght_mtd == 'mktcap':
            mktcap_df_copy = mktcap_df.reindex(index=stock_df.index)
            if any(np.isnan(mktcap_df_copy)):
                mktcap_df_copy = mktcap_df_copy.fillna(0)
            stock_df['MktCap'] = mktcap_df_copy[stock_df.index]
        elif wght_mtd == 'riskpar':
            stock_df['Volat'] = np.sqrt(pow(vol_df[stock_df.index], 2).sum(skipna=False))

        # Discard stocks for which I don't have a price (or, in case, a market cap. or volatility)
        if stock_df.isnull().values.any():
            stock_df = stock_df.dropna().copy()
            ret_df = ret_df[stock_df.index]

        # Final list of selected stocks
        tckrs_sel = stock_df.index

        # Compute the number of stocks to be traded by first computing the amount of wealth
        # to be allocated and then dividing by the price
        if wght_mtd.lower() == 'equal':  # Equally-weighted portfolio
            stock_df['Weights'] = [1 / tckrs_sel.size for _ in tckrs_sel]
        elif wght_mtd.lower() == 'metric':  # Metric (momentum or SEV)-weighted portfolio
            stock_df['Weights'] = stock_df['metric'] / stock_df['metric'].abs().sum()
        elif wght_mtd.lower() == 'riskpar':  # Risk-parity weighted portfolio
            stock_df['Weights'] = pow(stock_df['Volat'], -1) / pow(stock_df['Volat'], -1).sum()
        elif wght_mtd.lower() == 'mktcap':  # Metric (momentum or SEV)-weighted portfolio
            stock_df['Weights'] = stock_df['MktCap'] / stock_df['MktCap'].sum()
        elif wght_mtd.lower() == 'lotp' or wght_mtd.lower() == 'tp':  # Optimized portfolio
            allow_short_sell = (wght_mtd.lower() == 'tp')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="delta_grad == 0.0")
                warnings.filterwarnings("ignore", message="Singular Jacobian matrix")
                opt_weights = self.get_opt_weights(stock_df.index, ret_df, allow_short_sell,
                                                   risk_avers, max_lvrg, bema_ret_df)
                stock_df['Weights'] = opt_weights

        if stock_df.isnull().values.any():
            breakpoint()

        # Wealth to be allocated in the different assets
        stock_df['Quantity'] = self.val_tot_curr * stock_df['Weights']

        # Divide by the price (also accounting for transaction costs)
        stock_df['Quantity'] /= (trsctn_fee_fix + (1 + trsctn_fee_prop) * stock_df['Price'])

        # Apply floor() function to get an integer number of stocks
        stock_df['Quantity'] = np.floor(stock_df['Quantity'])

        return stock_df

    def compute_discr_returns(self, risk_free_ret):  # Discrete returns
        self.returns, self.log_returns, self.exc_ret, self.exc_ret_mean, self.volat, self.sharpe = \
            pf_analyz.compute_returns_volat_sharpe(self, risk_free_ret)

    def compute_sortino(self, min_acc_ret):  # Sortino ratio
        self.sortino = pf_analyz.compute_sortino(self, min_acc_ret)

    def compute_star(self, level):  # STAR ratio
        self.star = pf_analyz.compute_star(self, level)

    def compute_turnover(self):  # turnover
        self.turnover = pf_analyz.compute_turnover(self)

    def compute_alpha_beta(self, mkt_ret_df, risk_free_ret):  # alpha and beta factors
        self.alpha, self.beta, self.pvalues = pf_analyz.compute_alpha_beta(self, mkt_ret_df, risk_free_ret)

    def compute_drawdown(self):  # drawdown
        self.drawdown = pf_analyz.compute_drawdown(self)

    def analyze(self, mkt_ret_df, risk_free_ret=0, min_acc_ret=0, level=0.01):
        self.compute_discr_returns(risk_free_ret)
        self.compute_sortino(min_acc_ret=min_acc_ret)
        self.compute_star(level=level)
        self.compute_turnover()
        self.compute_alpha_beta(mkt_ret_df, risk_free_ret)
        self.compute_drawdown()
