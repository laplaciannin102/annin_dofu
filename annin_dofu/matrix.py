#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

行列関連modules.

@history:
    2021/10/13:
        初期版作成.
"""



# --------------------------------------------------------------------------------
# Load modules
# --------------------------------------------------------------------------------

import numpy as np



# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# all modules to import
__all__ = [
    'get_meshgrid_ndarray'
]



# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


def get_meshgrid_ndarray(n_dim, n_grid, x_range=[0, 1]):
    """
    get n-dimension grid data
    
    Args:
        n_dim: int
            n-dimension
        
        n_grid: int or list of int
            num of grid
        
        x_range: (list of )list of float
            x_min and x_max
    
    Returns:
        grid_nd_arr: ndarray
    
    Example:
        >> x_arr = get_meshgrid_ndarray(3, [1, 2, 3], [1, 3])
        >> print(x_arr)
        [[1. 1. 1.]
         [1. 1. 2.]
         [1. 1. 3.]
         [1. 3. 1.]
         [1. 3. 2.]
         [1. 3. 3.]]
    """
    
    if type(n_grid) == list:
        n_grid_list = n_grid
    else:
        n_grid_list = [n_grid for ii in range(n_dim)]
    
    if type(x_range[0]) == list:
        x_range_list = x_range
    else:
        x_range_list = [x_range for ii in range(n_dim)]
    
    # 引数の次元確認
    if (len(n_grid_list) != n_dim) or (len(n_grid_list) != n_dim):
        raise Exception('error: 引数の次元が不正です.')
    
    grid_1d_list = [
        np.linspace(x_range_list[ii][0], x_range_list[ii][1], num=n_grid_list[ii])
        for ii in range(n_dim)
    ]
    
    grid_nd_arr = [ii.reshape(-1) for ii in np.meshgrid(*grid_1d_list, copy=True)]
    grid_nd_arr = np.array(grid_nd_arr).T

    return grid_nd_arr

