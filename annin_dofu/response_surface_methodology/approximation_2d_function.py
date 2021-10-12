#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

関数近似関連modules.
2次近似.

@history:
    2021/10/13:
        初期版作成.
"""



# --------------------------------------------------------------------------------
# Load modules
# --------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fmin_bfgs



# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# all modules to import
__all__ = [
    'Approximation2dFunction'
]



# --------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------


class Approximation2dFunction():
    """
    2次近似.
    
    Parameters:

    Attributes:
        opt_A_arr: array-like
        opt_A_mat: array-like
        opt_cov:

    import scipy.optimize.curve_fit
    import numpy as np
    """
    
    def __init__(self):
        self.opt_A_arr = None
        self.opt_A_mat = None
        self.opt_cov = None
    
    
    def func2d(self, *args):
        """
        2次関数

        Args:
            X: array

            A: list of float
                係数
        """
        #print('args:', args)

        x_arr0 = np.array(args[0])
        A_arr0 = list(args[1:])

        n_dim = x_arr0.shape[0]
        n_samples =  x_arr0.T.shape[0]

        # 定数項分を増やす
        x_arr1 = np.vstack([np.ones(shape=[1, n_samples]), x_arr0])
        
        # 上三角行列にする
        A_arr1 = self.vec2triu(A_arr0, n_dim=n_dim+1)

        # (X^T)A(X)の計算
        y_arr = np.dot(A_arr1, x_arr1)
        y_arr = np.dot(x_arr1.T, y_arr)

        # 対角成分のみを抽出
        y_arr = np.diag(y_arr)

        return y_arr
    
    
    def n_tri_comp(self, n_dim):
        """
        上三角行列の0でない要素の数(の最大)を返す.

        Args:
            n_dim:
                dimension
        """
        return int((n_dim**2 + n_dim)/2)
    
    
    def vec2triu(self, vec, n_dim=1):
        """
        vectorを上三角行列にする.
        
        Args:
            vec: array-like
                vector
        """
        # 上三角行列にする
        tmp = []
        st = 0
        for i in range(n_dim):
            n = n_dim - i
            ed = st + n
            tmp += [list(np.zeros(shape=i)) + vec[st:ed]]
            st = ed
        triu_arr = np.array(tmp)
        return triu_arr
    
    
    def triu2vec(self, triu_arr):
        """
        上三角行列をベクトル(list)に変換する.転置されてしまうのでnpdarrayで入れること.

        Example:
            input: array[[1,2,3],[4,5,6],[7,8,9]]
            output: [1,2,3,5,6,9]
        """

        if type(triu_arr)==pd.DataFrame:
            triu_arr = np.array(triu_arr)

        n_dim = triu_arr.shape[0]
        vec = [triu_arr[ii][jj] for ii in range(n_dim) for jj in range(n_dim) if ii<=jj]
        return vec
    
    
    def fit(self, data_x, data_y):
        """
        フィッティングを行いモデルを作成する.
        
        Args:
            data_x: array-like
                features
            
            data_y: array-like
                target variable
        """
        x_arr = np.array(data_x)
        y_arr = np.array(data_y)
        
        n_rows = x_arr.shape[0]
        n_dim = x_arr.T.shape[0]

        if n_rows < self.n_tri_comp(n_dim+1):
            print('error: n_rows must be larger than n_tri_comp(n_dim+1)')
            return False
        
        self.opt_A_arr, self.opt_cov = curve_fit(f=self.func2d, xdata=x_arr.T, ydata=y_arr, p0=np.ones(shape=self.n_tri_comp(n_dim+1)), check_finite=True)
        self.opt_A_mat = self.vec2triu(list(self.opt_A_arr), n_dim=n_dim+1)
        return self
    
    
    def predict(self, data_x):
        """
        予測を行う.
        """
        x_arr = np.array(data_x)
        return self.func2d(x_arr.T, *self.opt_A_arr)
    
    
    def score(self, data_x, data_y, criterion='r2'):

        x_arr = np.array(data_x)
        y_arr = np.array(data_y)
        
        pred_y_arr = self.predict(x_arr)
        
        if criterion=='r2':
            from sklearn.metrics import r2_score
            score_val = r2_score(y_arr, pred_y_arr)
            
        elif criterion=='mae':
            from sklearn.metrics import mean_absolute_error as mae
            score_val = mae(y_arr, pred_y_arr)
        
        elif criterion=='mse':
            from sklearn.metrics import mean_squared_error as mse
            score_val = mse(y_arr, pred_y_arr)
            
        else:
            print('error: criterion is invalid.')

        return score_val

