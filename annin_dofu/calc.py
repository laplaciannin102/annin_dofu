#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

数値計算関連modules.

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



# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# all modules to import
__all__ = [
    'pro_round',
    'isin_section',
    'calc_integrate'
]



# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


def pro_round(num, ndigits=0):
    """
    数字を四捨五入で丸める.

    Args:
        num: int or float
            丸めたい数字.
        
        ndigits: int, optional(default=0)
            丸めた後の小数部分の桁数.
        
    Returns:
        rounded: int or float
            丸めた後の数字.
    """
    num *= 10 ** ndigits
    rounded = ( 2 * num + 1 ) // 2
    rounded /= 10 ** ndigits

    if ndigits == 0:
        rounded = int(rounded)

    return rounded


def isin_section(x, section, bounds=[False, True]):
    """
    xがsection(区間)に含まれているかどうかを返す.
    boundsがTrueだとClosed、FalseだとOpenになる.

    Args:
        x: int or float
            number
        
        section: list of float or list of str
            下端と上端.
        
        bounds: list of bool, optional(default=[False, True])
            左端と右端を含むかどうか.
            [左端を含むか, 右端を含むか]
    
    Returns:
        included_in_section: bool
            xがsectionに含まれているかどうか.
            含まれている時Trueを返す.
    """
    x = float(x)
    included_in_section = False

    # left and right
    left_bound = bounds[0]
    right_bound = bounds[1]

    # Lower and Upper
    lower = float(section[0])
    upper = float(section[1])

    if left_bound:
        left_condition = (lower <= x)
    else:
        left_condition = (lower < x)
    
    if right_bound:
        right_condition = (x <= upper)
    else:
        right_condition = (x < upper)
    
    if left_condition and right_condition:
        included_in_section = True
    
    return included_in_section


def calc_integrate(func=(lambda x: x), a=0, b=1, mode='quad', num=100):
    """
    積分を計算する(calc integrate).
    関数, 上端, 下端, 計算方法, 分割数を指定する.
    
    Args:
        func: function
        
        a: float, optional(default=0)
            lower end.
        
        b: float, optional(default=1)
            upper end.
        
        mode: str, optional(default='quad')
            'quad', 'trapz', 'simps'
        
        num: int
            the number of divisions
    
    Returns:
        area: float
            area
    
    Examples:
        # integrate sample
        # square func
        f = (lambda x: x**2)
        a = -1
        b = 2

        print('*' * 80)
        print('func: {}, a: {}, b: {}'.format('square func', '-1', '2'))
        print('*' * 80)

        for mode in ['quad', 'trapz', 'simps']:
            area = calc_integrate(f, a, b, mode=mode)
            print('mode: {}'.format(mode))
            print('area: {}'.format(area))
            print('*' * 80)

        print()

        # normal dist
        f = scipy.stats.norm(loc=0, scale=1).pdf
        a = - np.inf
        b = - a

        print('*' * 80)
        print('func: {}, a: {}, b: {}'.format('normal dist', '-inf', 'inf'))
        print('*' * 80)

        for mode in ['quad', 'trapz', 'simps']:
            area = calc_integrate(f, a, b, mode=mode, num=10000)
            print('mode: {}'.format(mode))
            print('area: {}'.format(area))
            print('*' * 80)

        print()

        # lower and upper
        a = - 1e+4
        b = - a

        print('*' * 80)
        print('func: {}, a: {}, b: {}'.format('normal dist', '-1e+4', '1e+4'))
        print('*' * 80)

        for mode in ['quad', 'trapz', 'simps']:
            area = calc_integrate(f, a, b, mode=mode, num=10000)
            print('mode: {}'.format(mode))
            print('area: {}'.format(area))
            print('*' * 80)
    """

    if mode=='quad':
        # gaussian quadrature(Fortran のライブラリ QUADPACK)
        area, err = scipy.integrate.quad(func, a, b)
    
    else:
        
        x_arr = np.linspace(a, b, num+1)
        y_arr = np.array([func(x) for x in x_arr])

        if mode=='trapz':
            # 台形公式(台形は英語でtrapezoid)
            area = scipy.integrate.trapz(y_arr, x_arr)

        elif mode=='simps':
            # シンプソンの公式(Simpson's rule)
            area = scipy.integrate.simps(y_arr, x_arr)

        else:
            print('error: mode error')
            area = False
    
    return area


