#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

統計関連modules.

@history:
    2021/10/13:
        初期版作成.
"""



# --------------------------------------------------------------------------------
# Load modules
# --------------------------------------------------------------------------------

import sys, os
import gc
import numpy as np
import scipy
import statsmodels.api as sm

# original modules
# from .utils import *
from .calc import *



# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# all modules to import
__all__ = [
    'sturges_rule',
    'norm_dist_prob',
    'sp_norm_dist_prob',
    'confidence_interval'
]



# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


def sturges_rule(num):
    """
    スタージェスの公式を用いて,
    サンプルサイズから適切な階級(カテゴリ, ビン(bins))の数を計算する.

    Args:
        num: int
            サンプルサイズ. 原則1以上の整数を想定.
    
    Returns:
        n_bins: int
            スタージェスの公式から導かれた適切な階級の数.
    
    TeX notion:
        \[bins = 1 + \log_2{N} \nonumber \]
    """
    # numが0以下の時は1を返す
    if num <= 0:
        num = 1
        return 1
    
    # スタージェスの公式
    n_bins = int(pro_round(1 + np.log2(num), 0))
    
    return n_bins


def norm_dist_prob(x=0, mu=0, sigma=1):
    """
    正規分布の確率密度関数(probability density function normal distribution).

    Args:
        x: float, optional(default=0)
            random variable

        mu: float, optional(default=0)
            mean value

        sigma: float, optional(default=0)
            standard deviation
    
    Returns:
        y: float
            pdf value
    
    Example:
        x_arr = np.linspace(-5, 5, 1000)
        y_arr = norm_dist_prob(x_arr)

        plt.figure(facecolor='white')
        plt.plot(x_arr, y_arr)
        plt.title('normal distribution')
        plt.grid()
        plt.show()
    
    TeX notion:
        $$
        \begin{aligned}
        \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)
        \end{aligned}
        $$
    """
    y = (1 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(-((x - mu)**2 / (2 * (sigma**2))))
    return y


def sp_norm_dist_prob(x=0, mu=0, sigma=1):
    """
    正規分布の確率密度関数(probability density function normal distribution).
    scipyバージョン.

    Args:
        x: float, optional(default=0)
            random variable

        mu: float, optional(default=0)
            mean value

        sigma: float, optional(default=0)
            standard deviation
    
    Returns:
        y: float
            pdf value
    """
    # scipy
    y = scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return y


def confidence_interval(arr, alpha=0.95):
    """
    母平均の信頼区間(confidence interval).

    Args:
        arr: array-like
        
        alpha: float, optional(default=0.95)
            0.95の時, 95%信頼区間
    
    Returns:
        interval: list
            [lower, upper]
            lower endpoint and upper endpoint
    """
    # import scipy
    # import statsmodels.api as sm
    
    if len(arr) <= 1:
        # return [-1e+10, 1e+10]
        return [np.nan, np.nan]
    
    # 平均値(mean)
    mean_val = np.mean(arr)

    # 標本平均の標準誤差(standard error of the mean; SEM)
    sem_val = scipy.stats.sem(arr)

    # 自由度
    deg_of_free = len(arr) - 1

    interval = list(
        scipy.stats.t.interval(
            alpha=alpha,
            df=deg_of_free,
            loc=mean_val,
            scale=sem_val
        )
    )
    
    return interval


def temp_multivar_normal_dist_sampling(n_dim, n_size, mean_val=0, var_val=1):
    """
    多次元正規分布の簡易的なサンプリングを行う.
    各変数は独立, 平均と分散は共通のものを使用する.
    np.random.multivariate_normalを使用.

    Args:
        n_dim : int
            多次元正規分布の次元
        
        n_size: array-like

        mean_val: float, optional(default=0)
            平均値
        
        var_val: float, optional(default=1)
            分散

    Returns:
        out: ndarray
    """
    import numpy as np
    x_mean = [mean_val for ii in range(n_dim)]
    #x_cov = np.ones((n_dim, n_dim)) * var_val
    x_cov = np.eye(n_dim) * var_val
    return np.random.multivariate_normal(mean=x_mean, cov=x_cov, size=n_size)


# TODO: 偏相関行列の算出
"""
def partialcorr(data):
    pass
    # return pcorr_arr
"""

