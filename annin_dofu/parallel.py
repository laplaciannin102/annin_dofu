#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

並列処理関連modules.

@history:
    2021/10/13:
        初期版作成.
"""



# --------------------------------------------------------------------------------
# Load modules
# --------------------------------------------------------------------------------

from multiprocess import Pool
from joblib import Parallel, delayed

# original modules
# from .utils import *



# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# all modules to import
__all__ = [
    'multiprocess_func',
    'parallel_func'
]



# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


# from multiprocess import Pool
def multiprocess_func(func, args_list, processes=None, initializer=None, initargs=(), maxtasksperchild=None, chunksize=None):
    """
    マルチプロセスで並列処理を行う.
    
    Args:
        func: function
            並列処理したい関数.
        
        args_list: list
            関数の引数リスト.
        
        processes: int or None, optional(default=None)
            プロセス数.
    
    Returns:
        ret_list: list
            list of Returns
    
    Example:
        
        def func0(x, y):
            import time
            time.sleep(5)
            return [x+y, x-y]

        args_list0 = [
            [1, 1],
            [2, 2],
            [3, -1],
            [4, -2]
        ]

        ret0 = multiprocess_func(
            func = func0,
            args_list = args_list0,
            processes = 4
        )
        print('list of Returns:', ret0)
        # list of Returns: [[2, 0], [4, 0], [2, 4], [2, 6]]
    
    Remark:
        Multiprocessing example giving AttributeError
        (https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror)
        
        How to use multiprocessing pool.map with multiple arguments?
        (https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments)
    """
    with Pool(
        processes=processes,
        initializer=initializer,
        initargs=initargs,
        maxtasksperchild=maxtasksperchild
    ) as pool_obj:
        
        ret_list = pool_obj.starmap(func=func, iterable=args_list, chunksize=chunksize)
    
    return ret_list


# from joblib import Parallel, delayed
def parallel_func(func, args_list, args_type='list', n_jobs=-1, verbose=1, backend='loky', prefer='threads', batch_size='auto'):
    """
    execute function in parallel.
    joblibを使用. joblib内含め, multiprocessing moduleのエラーにより, multi process非対応, multi threadsのみ対応.
    
    Args:
        func: function
            並列処理したい関数.
        
        args_list: list
            関数の引数リスト.
        
        args_type: str, optional(default='list')
            'list' or 'dict'
            引数の型.
        
        n_jobs: int, optional(default=-1)
            worker数.
        
        verbose: int, optional(default=0)
            min:0, max:50
            1以上の時, 途中経過が表示される.
        
        backend: str, optional(default='loky')
            'loky' or 'multiprocessing' or 'threading'
            'multiprocessing'はerrorが出やすい.
            並列処理の方法. マルチスレッドかマルチプロセス.
        
        prefer: str, optional(default='threads')
            'threads' or 'processes'
            並列処理の方法. マルチスレッドかマルチプロセス.
        
        batch_size: int or 'auto', optional(default='auto')
            バッチサイズ.
            同時に処理する処理数.
    
    Returns:
        ret_list: list
            list of Returns
    
    Examples:
        
        def func0(x, y):
            import time
            time.sleep(5)
            return [x+y, x-y]

        args_list0 = [
            [1, 1],
            [2, 2],
            [3, -1],
            [4, -2]
        ]

        args_list1 = [
            {'x': 1, 'y': 1},
            {'x': 2, 'y': 2},
            {'x': 3, 'y': -1},
            {'x': 4, 'y': -2}
        ]

        ret0 = parallel_func(func=func0, args_list=args_list0, args_type='list', verbose=1)
        print('type of args is list:', ret0)
        # type of args is list: [[2, 0], [4, 0], [2, 4], [2, 6]]

        ret1 = parallel_func(func=func0, args_list=args_list1, args_type='dict', verbose=2, prefer='processes')
        print('type of args is dict:', ret1)
        # type of args is dict: [[2, 0], [4, 0], [2, 4], [2, 6]]
    """
    ret_list = []
    
    if args_type=='list':
        
        ret_list = Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
            backend=backend,
            prefer=prefer,
            batch_size=batch_size
        )([delayed(func)(*args) for args in args_list])
    
    elif args_type=='dict':
        ret_list = Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
            backend=backend,
            prefer=prefer,
            batch_size=batch_size
        )([delayed(func)(**args) for args in args_list])
    
    return ret_list

