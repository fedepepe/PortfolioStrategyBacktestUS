## Description 
This project implements and backtests a long-only portfolio strategy based on a two-step procedure: in the first one, the algorithm screens the stock data in a rolling window to select a user-defined number of optimal ones, and, in the second, computes the portfolio allocation, using either Markowitz' mean-variance analysis or empirics. In this way, a potentially very large asset universe is shrunk to a few tens of assets, which dramatically reduces the estimation error of the covariance matrix of asset returns.

I perform the asset selection by recasting the Markowitz’ risk minimization problem as a regression problem, as explained in:
Fan, J. and Zhang, J. and Yu, K. (2021). Vast Portfolio Selection with Gross-Exposure Constraints. Journal of the American Statistical Association, 107(498):592-606. This enables us to access the vast statistical toolbox associated to regression.

I build upon the approach presented in: Dan Wang, C. and Chen, Z. and Lian Y. and Chen M. (2020). Asset Selection based on High Frequency Sharpe Ratio.
Journal of Econometrics, and I further enhance it by adopting the improved covariance matrix estimator, tested in the code in the ImprovingMVPortfolio repository.

The performance of the strategy is compared against the widely-used momentum strategy, its risk-managed counterpart and Dan Wang's method, which are all implemented in the code. The momentum strategy is presented in: Jegadeesh, N. and Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. Journal of Finance, 48:65–91. For risk-managed momentum, please refer to: Barroso, P. and Santa-Clara, P. (2015). Momentum Has Its Moments. Journal of Financial Economics, 116(1):111–120.

The dataset consists of daily closing prices and realized volatilities observed for U.S. stocks listed in the SP500 index in the time frame spanning between 1/1/2018 and 25/8/2020, including the market collapse due to COVID-19 pandemic.

## Back-testing parameters  
They are located in the main() method in main.py

endow: (optional) Initial amount of money to be invested. Float  

n_stk: Number of stocks to hold in the portfolio. Integer  

n_obs_ar: Lengths of rolling window (in trading days). Array of integers  

n_reb_ar: Portfolio rebalancing interval (in trading days). Array of integers  

algo: Algorithm to use for stock selection. String. Possible choices are: 'mtm', 'rmmtm', 'sev' or 'sev+'  
* mtm:   Momentum. Pick stocks with largest value of average return  
* rmmtm: Risk-managed Momentum. Pick stocks with largest value of average Sharpe ratio  
* sev:   Dan Wang's stock selection method  
* sev+:  Improved Dan Wang's method  

wght_mtd: Stock weighting methods for portfolio allocation. List of strings. Possible choices are: 'equal', 'metric', 'mktcap', 'riskpar', 'lotp', 'tp'  
* equal:   Equally-weighted portfolio  
* metric:  Weights are proportional to momentum or SEV  
* mktcap:  Weights are proportional to market capitalization  
* riskpar: Risk-parity portfolio  
* tp:      Tangency Portfolio (obtained via mean-variance analysis)  
* lotp:    Long-Only Tangency Portfolio (obtained via mean-variance analysis)  

trsctn_fee_fix:  (optional) Fixed transaction fees  

trsctn_fee_prop: (optional) Proportional transaction fees  
