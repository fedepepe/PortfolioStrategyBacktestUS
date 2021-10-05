#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:12:38 2021

@author: federico
"""

import numpy as np
from sklearn.model_selection import LeaveOneOut
import warnings
import sys


def gau_ker(x):
    return 1. / np.sqrt(2 * np.pi) * np.exp(- 0.5 * pow(x, 2))


def kernel_density(x, x_pts, bw):
    # x:     vector of input samples
    # x_pts: vector of values where the density is to be calculated
    # bw:    kernel bandwidth
    n_obs = len(x)
    fx_hat = np.nan * x_pts
    for i in np.arange(len(x_pts)):
        ker_x = 1. / bw * gau_ker((x_pts[i] - x) / bw)
        fx_hat[i] = 1. / n_obs * sum(ker_x)
    return fx_hat


def kernel_density_support(x, bw):
    n_obs = len(x)
    gridsize = int(2 ** np.ceil(np.log2(10. * n_obs)))
    cut = 3.0
    a = np.min(x) - cut * bw
    b = np.max(x) + cut * bw
    x_pts, dx = np.linspace(a, b, gridsize, retstep=True)
    return x_pts, dx


def get_bw_scott(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Get a robust estimate of sigma
        sig = np.nanmedian(np.abs(x - np.nanmedian(x))) / 0.6745

    if sig <= 0:
        sig = max(x) - min(x)

    if sig > 0:
        n = len(x)
        return 1.059 * sig * n ** (-0.2)
    else:
        return np.nan


def get_optimum_cv_bandwidth(x, bw_ref, n_bw_points=21):
    bw_vec = np.logspace(np.log10(bw_ref / 10.), np.log10(bw_ref * 10.), n_bw_points)
    n_obs = len(x)

    emp_risk = np.array([0.0] * len(bw_vec))
    emp_risk_1 = np.array([0.0] * len(bw_vec))
    emp_risk_2 = np.array([0.0] * len(bw_vec))

    i = 0
    for bw in bw_vec:
        x_pts, dx = kernel_density_support(x, bw)
        fx_hat = kernel_density(x, x_pts, bw)
        emp_risk_1[i] = sum(pow(fx_hat, 2)) * dx

        # Cross-validation to find optimal bandwidth
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(x):
            x_train, x_test = x[train_index], x[test_index]
            fx_hat = kernel_density(x_train, x_test, bw)
            emp_risk_2[i] = emp_risk_2[i] + fx_hat

        # Compute empirical risk
        emp_risk[i] = emp_risk_1[i] - 2. / n_obs * emp_risk_2[i]
        i += 1

    # Select bandwidth associated to minimum empirical risk
    bw_opt = bw_vec[emp_risk.argmin()]
    return bw_opt


def compute_sev(x, y, cv_opt_bw=False):
    bw_scott = get_bw_scott(x)  # Scott's Rule of Thumb

    if np.isnan(bw_scott):
        return np.nan

    if cv_opt_bw:
        bw = get_optimum_cv_bandwidth(x, bw_scott)
    else:
        bw = bw_scott

    # Compute the support where the density of x is to be estimated
    x_pts, dx = kernel_density_support(x, bw)

    n_obs, n_pts = len(x), len(x_pts)

    fx_hat, phihat = np.nan * x_pts, np.nan * x_pts
    for i in np.arange(n_pts):
        ker_x = 1. / bw * gau_ker((x_pts[i] - x) / bw)
        fx_hat[i] = 1. / n_obs * sum(ker_x)
        phihat[i] = 1. / n_obs * sum(ker_x * y)

    fx_hat[fx_hat == 0] = sys.float_info.min  # Replace zeros with minimum float number

    # Conditional expectation of Y given X, i.e., regression function
    # a.k.a. Nadaraya-Watson estimator
    r_hat = phihat / fx_hat

    # # Local kernel regression (Nadaraya-Watson estimator)
    # # explaining var. x is sr_df[stk], explained var. y is hsr_idx
    # ker_reg = KernelReg(y, x, var_type='c', reg_type='lc', bw=[bw])
    # r_hat = ker_reg.fit(x_pts)[0] # col [1] stores the derivatives

    sev = sum(pow(r_hat, 2) * fx_hat) * dx - pow(y.mean(), 2)
    sev = float(sev / y.var(ddof=1))
    return sev


def chunks(lst, n):
    # Yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def compute_sev_multiproc(d, x_df, y_df, cv_opt_bw=False):
    # Suppress warning message caused by NaNs in the dataframes for kernel regr.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")

        for tkr in x_df.columns:
            sev = compute_sev(x_df[tkr].values, y_df.values, cv_opt_bw)
            d[x_df[tkr].name] = sev
