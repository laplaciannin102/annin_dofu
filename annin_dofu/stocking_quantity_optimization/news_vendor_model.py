#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

News Vendor Model関連modules.

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

# original modules
from ..calc import *
from ..stats import *



# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# all modules to import
__all__ = [
    'calc_demand_diff_ratio_from_pred_demand',
    'calc_quantity_diff_ratio_from_pred_demand',
    'calc_gross_profit',
    'calc_expected_gross_profit',
    'calc_best_excess',
    'calc_best_expected_gross_profit'
]



# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


def calc_demand_diff_ratio_from_pred_demand(demand=None, pred_demand=None):
    """
    calc demand_diff_ratio_from_pred_demand
    
    Args:
        demand: float or array-like of float
            demand
        
        pred_demand: float or array-like of float
            demand
    
    Returns:
        alpha: float or array-like of float
            demand_diff_ratio_from_pred_demand
    """
    try:
    
        # if (type(demand)==list)or(type(demand)==np.ndarray):
        if type(demand) in [list, np.ndarray]:

            demand_arr = np.array(demand)
            pred_demand_arr = np.array(pred_demand)
            
            alpha = (demand_arr - pred_demand_arr) / pred_demand_arr

            return 

        else:

            alpha = (demand - pred_demand) / pred_demand

            return alpha
    
    except Exception as e:
        print('error:')
        print(e)


def calc_quantity_diff_ratio_from_pred_demand(quantity=None, pred_demand=None):
    """
    calc quantity_diff_ratio_from_pred_demand
    
    Args:
        quantity: float or array-like of float
            quantity
        
        pred_demand: float or array-like of float
            quantity
    
    Returns:
        _x: float or array-like of float
            quantity_diff_ratio_from_pred_demand
    """
    try:
    
        # if (type(quantity)==list)or(type(quantity)==np.ndarray):
        if type(quantity) in [list, np.ndarray]:

            quantity_arr = np.array(quantity)
            pred_demand_arr = np.array(pred_demand)
            
            _x = (quantity_arr - pred_demand_arr) / pred_demand_arr

            return 

        else:

            _x = (quantity - pred_demand) / pred_demand

            return _x
    
    except Exception as e:
        print('error:')
        print(e)


def calc_gross_profit(demand=None, sales=None, quantity=None, price=None, cost=None):
    """
    calc gross profit
    
    Args:
        demand: float or array-like of float
            demand
        
        sales: float or array-like of float
            sales
        
        quantity: float or array-like of float
            stocking quantity
        
        price: float or array-like of float
            price
        
        cost: float or array-like of float
            cost
    
    Returns:
        gross_profit: float or array-like of float
            gross profit
    """
        
    if quantity is None:
        print('args error: quantity is necessary.')
        return False
    
    try:
    
        # if (type(quantity)==list)or(type(quantity)==np.ndarray):
        if type(quantity) in [list, np.ndarray]:

            quantity_arr = np.array(quantity)

            if not sales is None:

                sales_arr = np.array(sales)
                gross_profit = price * sales_arr - cost * quantity_arr

            else:

                demand_arr = np.array(demand)
                sales_arr = np.min([quantity_arr, demand_arr], axis=0)

                gross_profit = price * sales_arr - cost * quantity_arr

            return gross_profit

        else:

            gross_profit = price * sales - cost * quantity

            return gross_profit
    
    except Exception as e:
        print('error:')
        print(e)


def calc_expected_gross_profit(x, alpha_density_func, pred_demand, price, cost):
    """
    calc expected gross profit.
    
    Args:
        x: float
            quantity diff ratio from pred demand.
        
        alpha_density_func: function
            alpha density function.
            alpha is demand diff ratio from pred demand.
        
        pred_demand: float or array-like of float
            demand.
        
        price: float or array-like of float
            price.
        
        cost: float or array-like of float
            cost.
    
    Returns:
        expected_gross_profit: float or array-like of float
            expected gross profit
    """
    try:
    
        integrand_func = (lambda u: (u - x) * alpha_density_func(u))

        expected_gross_profit = pred_demand * (
            price * (calc_integrate(func=integrand_func, a=-np.inf, b=x)) + (price - cost) * (x + 1)
        )

        return expected_gross_profit

    except Exception as e:
        print('error:')
        print(e)


def calc_best_excess(
    x_min=-1,
    x_max=2,
    x_step=1e-3,
    alpha_cumulative_dist_func=None,
    alpha_density_func=None,
    price=None,
    cost=None,
    epsilon=1e-2
):
    """
    calc best excess stocking quantity.
    
    Args:
        x_min: float, optional(default=-1)
            min of x
        
        x_max: float, optional(default=2)
            max of x
        
        x_step: float, optional(default=1e-3)
            step of x
        
        alpha_cumulative_dist_func: function
            alpha cumulative distribution function.
            alpha is demand diff ratio from pred demand.
        
        alpha_density_func: function
            alpha probablity density function.
            alpha is demand diff ratio from pred demand.
        
        price: float or array-like of float
            price.
        
        cost: float or array-like of float
            cost.
        
        epsilon: float, optional(default=1e-2)
    
    Returns:
        best_x: float
            best_x
    """
    try:
        
        x_arr = np.arange(x_min, x_max + x_step, x_step)
        profit_ratio = (price - cost) / price
        
        result_list = []
        best_x = x_min
        
        for x in x_arr:
            
            if alpha_cumulative_dist_func is None:
                area_val = calc_integrate(alpha_density_func, -np.inf, x, mode='quad')
            
            else:
                area_val = alpha_cumulative_dist_func(x)
            
            # 絶対残差
            abs_res = np.abs(profit_ratio - area_val)
            
            result_list += [[x, abs_res]]
            
            if abs_res <= epsilon:
                best_x = x
                break
        
        result_arr = np.array(result_list)
        best_x = result_arr[result_arr.argmin(axis=0)[1]][0]
        
        return best_x

    except Exception as e:
        print('error:')
        print(e)


def calc_best_expected_gross_profit(best_x, alpha_density_func, pred_demand, price, cost):
    """
    calc best expected gross profit.
    
    Args:
        best_x: float
            quantity diff ratio from pred demand.
        
        alpha_density_func: function
            alpha density function.
            alpha is demand diff ratio from pred demand.
        
        pred_demand: float or array-like of float
            demand.
        
        price: float or array-like of float
            price.
        
        cost: float or array-like of float
            cost.
    
    Returns:
        best_expected_gross_profit: float or array-like of float
            expected gross profit
    """
    try:
    
        integrand_func = (lambda u: u * alpha_density_func(u))

        best_expected_gross_profit = pred_demand * (
            price * (calc_integrate(func=integrand_func, a=-np.inf, b=best_x)) + price - cost
        )

        return best_expected_gross_profit

    except Exception as e:
        print('error:')
        print(e)


