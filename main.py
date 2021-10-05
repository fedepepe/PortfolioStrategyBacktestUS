# This project implements and backtests a long-only portfolio strategy based on a two-step procedure: in the first one,
# the algorithm screens the stock data in a rolling window to select a user-defined number of optimal ones, and,
# in the second, computes the portfolio allocation, using either Markowitz' mean-variance analysis or empirics.
# In this way, a potentially very large asset universe is shrunk to a few tens of assets, which dramatically
# reduces the estimation error of the covariance matrix of asset returns.

# I perform the asset selection by recasting the Markowitz’ risk minimization problem as a regression problem,
# as explained in:
# Fan, J. and Zhang, J. and Yu, K. (2021). Vast Portfolio Selection with Gross-Exposure Constraints.
# Journal of the American Statistical Association, 107(498):592-606.
# This enables to access the vast statistical toolbox associated to regression.

# I build upon the approach presented in:
# Dan Wang, C. and Chen, Z. and Lian Y. and Chen M. (2020). Asset Selection based on High Frequency Sharpe Ratio.
# Journal of Econometrics
# and I further enhance it by adopting the improved covariance matrix estimator, tested in the ImprovingMVPortfolio
# repository

# The performance of the strategy is compared against other the widely used momentum strategy, its risk-managed
# counterpart and Dan Wang's method, which are all implemented in the code.
# The momentum strategy is presented in:
#   Jegadeesh, N. and Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for
#   Stock Market Efficiency. Journal of Finance, 48:65–91.
# For risk-managed momentum, please refer to:
#   Barroso, P. and Santa-Clara, P. (2015). Momentum Has Its Moments. Journal of Financial Economics, 116(1):111–120.

# The dataset consists of daily closing prices and realized volatilities observed for U.S. stocks listed in the
# SP500 index in the time frame spanning between 1/1/2018 and 25/8/2020, including the market collapse due to
# COVID-19 pandemic

import matplotlib.pyplot as plt


def main():
    import numpy as np
    import pandas as pd
    from portfolio_sim import PortfolioSimulator
    import portfolio_plot as pf_plot
    import time

    # Load dataset of daily stock closing prices, returns, volatilities and market capitalization
    [last_price_df, return_df, real_vol_df, mktcap_df, rf_ret_df] = pd.read_pickle('SP500_thesis.pkl')

    last_price_df = last_price_df[last_price_df.columns.intersection(mktcap_df.columns)]
    return_df = return_df[return_df.columns.intersection(mktcap_df.columns)]
    real_vol_df = real_vol_df[real_vol_df.columns.intersection(mktcap_df.columns)]

    mkt_ret_df, mkt_idx_df = pd.read_pickle('mkt_idx.pkl')

    # %% Set parameters and initialize
    endow = 1e6  # Initial amount of money to be invested
    n_stk = 30  # Number of stocks to hold in the portfolio
    n_obs_ar = np.arange(5, 65, 5)  # Length of rolling window (in trading days)
    n_reb_ar = np.arange(5, 27, 2)  # Portfolio rebalancing interval (in trading days)

    # Algorithm to use for stock selection
    # Possible choices are: mtm, rmmtm, sev or sev+
    # mtm:   Momentum. Pick stocks with largest value of average return
    # rmmtm: Risk-managed Momentum. Pick stocks with largest value of average Sharpe ratio
    # sev:   Dan Wang's stock selection method
    # sev+:  Improved Dan Wang's method
    algo = 'sev+'

    # Stock weighting for portfolio allocation
    # Possible choices are: equal, metric, mktcap, riskpar, lotp and/or tp
    # equal:   Equally-weighted portfolio
    # metric:  Weights are proportional to momentum or SEV
    # mktcap:  Weights are proportional to market capitalization
    # riskpar: Risk-parity portfolio
    # tp:      Tangency Portfolio (obtained via mean-variance analysis)
    # lotp:    Long-Only Tangency Portfolio (obtained via mean-variance analysis)
    wght_mtd = ['equal', 'metric', 'mktcap', 'riskpar', 'lotp', 'tp']  # Weighting method for stock alloc.

    # Transaction costs
    trsctn_fee_fix = 0  # Fixed transaction fees
    trsctn_fee_prop = 10e-4  # Proportional transaction fees

    # %% Simulate
    results_dir, results_tag = './' + '/results/', 'thesis_sw_' + f'{n_stk}'

    portfolio_sim = PortfolioSimulator(n_stk=n_stk, n_obs=n_obs_ar[0], n_reb=n_reb_ar[0],
                                       algo=algo, wght_mtd=wght_mtd, last_price_df=last_price_df,
                                       return_df=return_df, volat_df=real_vol_df, mktcap_df=mktcap_df,
                                       bema_ret_df=rf_ret_df, endow=endow, idx_start=max(n_obs_ar), lag=1,
                                       trsctn_fee_fix=trsctn_fee_fix, trsctn_fee_prop=trsctn_fee_prop,
                                       risk_avers_factor=None,
                                       multi_proc=True, cv_opt_bw=False,
                                       results_dir=results_dir, results_tag=results_tag,
                                       save_stk_hist=False)

    start = time.time()

    # Main loop
    n_sim = len(n_obs_ar) * len(n_reb_ar)
    n_done = 0
    for n_obs in n_obs_ar:
        for n_reb in n_reb_ar:
            portfolio_sim.n_obs = n_obs
            portfolio_sim.n_reb = n_reb

            print(f'\n --- Running sim. {n_done + 1} of {n_sim} ---')

            portfolio_sim.backtest()

            end_curr = time.time()
            elapsed = end_curr - start  # Total time elapsed

            # Analyze portfolio performance
            portfolio_sim.analyze(mkt_ret_df=mkt_ret_df, risk_free_ret=rf_ret_df)

            # Print results to text file
            portfolio_sim.print_results()

            n_done += 1
            elapsed_mean = elapsed / n_done

            print('\n --- Total time elapsed: {0} (average sim. time: {1}) --- \n'.format(
                time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed)),
                time.strftime('%Mm %Ss', time.gmtime(elapsed_mean))))

    # %% Plot
    for wm in wght_mtd:
        results_df = portfolio_sim.load_results(wght_mtd=wm, filename=None)
        pf_plot.plot_heatmap(results_df, 'alpha', descr=algo + ', ' + wm)

    plt.show()


if __name__ == '__main__':
    main()
