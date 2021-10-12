#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

関数近似関連modules.
RBF補間近似.

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
    'ApproximationRBFInterpolation'
]



# --------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------


class ApproximationRBFInterpolation():
    """
    RBF(放射基底関数, Radial Basis Function)による補間を利用した関数近似モデル.

        Attributes:
            rbf_wopt:
            is_trained:
            params: dict
                rbf: str, optional(default='gaussian')
                    radial basis function
                
                w0_arr: array-like, optional(default=None)
                    initial weight value
                
                beta_val: float, optional(default=1)

                lambda_val: float, optional(default=1)
                
                n_divide: float, optional(default=1)
                
                gtol: float, optional(default=1e-8)

                n_samples: int, optional(default=100)
                    number of center points of RBF
                
                rel_ct: str, optional(default='equal')
                    relationship of center points and train data
                    'equal': center == train
                    'all': train is all of input train data
                    'exclusive': train and center are exclusive
                
                sampling_method: str, optional(defaulot='random')
                    'random', 'last'
                
                random_state: int, optional(default=57)
                    random seed
                
                hyper_opt: bool, optional(default=False)
                    ifelse optimize hyper parameter
                
                opt_disp: bool, optional(default=False)

                print_debug: bool, optional(default=False)

            debug_log:
    
    import scipy.optimize.fmin_bfgs
    """
    
    def __init__(
        self,
        distance='euclidian',
        rbf='gaussian',
        w0_arr=None,
        beta_val=1,
        lambda_val=1,
        n_divide=1,
        gtol=1e-8,
        _eps=1e-8,
        n_samples=100,
        rel_ct='equal',
        sampling_method='random',
        random_state=57,
        hyper_opt=False,
        opt_disp=False,
        print_debug=False
        ):
        """
        コンストラクタ

        Args:
            rbf: str
                radial basis function
            
            w0_arr: array-like:
                initial weights
            
            beta_val: float

            lambda_val: float

            n_samples: int, optional(default=100)
        """

        # 重みの最適値
        self.rbf_wopt = None

        # 基底の中心
        self.center_x_arr = None
        self.center_y_arr = None

        # 基底の中心セットフラグ
        self.center_is_set = False

        # 基底の中心がsamplingされたものであるフラグ
        self.center_is_sampled = False

        # 学習済フラグ
        self.is_trained = False

        # 距離関数
        # 初期はEuclidian distance
        # self.distance_func = self.euclidian_distance
        # self.squared_distance_func = self.squared_euclidian_distance
        
        # norm_funcに名称変更 at 2020/04/14
        self.norm_func = self.euclidian_distance
        self.squared_norm_func = self.squared_euclidian_distance # normの2乗

        # RBF(放射基底関数)
        # 初期はgaussian RBF
        self.radial_basis_func = self.gaussian_rbf
        
        # 引数rbfに応じてrbfをセットする
        self.set_rbf(rbf_text=str(rbf))

        # epsilon
        # logをなどに使用
        self._eps = _eps

        # パラメータ
        self.params = {}
        self.debug_log = ''
        
        # 重みの初期値
        if w0_arr is None:

            """
            # 指定が無ければ1埋めする
            w0_arr = np.ones(shape=n_samples)
            """
            # 指定が無ければ0埋めする
            w0_arr = np.zeros(shape=n_samples)
        
        params_dict = {
            'distance': distance,
            'rbf': rbf,
            'w0_arr': w0_arr,
            'beta_val': beta_val,
            'lambda_val': lambda_val,
            'n_divide': n_divide,
            'gtol': gtol,
            'n_samples': n_samples,
            'rel_ct': rel_ct,
            'sampling_method': sampling_method,
            'random_state': random_state,
            'hyper_opt': hyper_opt,
            'opt_disp': opt_disp,
            'print_debug': print_debug
        }
        
        # self.params = params_dict
        self.set_params(**params_dict)
    
    
    def set_params(self, **params):
        """
        set parameters

            Args:
                params: dict
                    parameters
            
            Returns:
                self: object
                    model
        """
        keys = list(params.keys())
        self_keys = list(self.params.keys())

        # n_samplesが変更になる場合は重み初期値を変更しておく
        if ('n_samples' in keys)and('n_samples' in self_keys):
            if (params['n_samples']!=self.params['n_samples']):
                self.params['w0_arr'] = np.zeros(shape=params['n_samples'])
        
        for key in keys:
            self.params[key] = params[key]

        return self
    
    
    def get_params(self, deep=False):
        """
        get parameters

            Returns:
                params: dict
                    parameters
        """
        return self.params
    

    def set_center_points(self, center_x, center_y):
        """
        set samples for center points of RBF

        Args:
            center_x: array-like

            center_y: array-like
        """
        center_x_arr = np.array(center_x)
        center_y_arr = np.array(center_y)

        self.center_x_arr = center_x_arr
        self.center_y_arr = center_y_arr

        # 中心セットフラグをTrueに
        self.center_is_set = True
    

    def set_rbf(self, rbf_text='gaussian'):
        """
        set radial basis function

        Args:
            rbf_text: str, optional(default='gaussian')
        """

        rbf_text = str(rbf_text)

        if rbf_text == 'gaussian':
            self.radial_basis_func = self.gaussian_rbf
        
        elif rbf_text == 'TPS':
            self.radial_basis_func = self.thin_plate_spline_rbf
    

    def squared_euclidian_distance(self, x1, x2):
        """
        squared euclidian distance

        Args:
            x1: array-like
                vector1

            x2: array-like
                vector2
        
        Returns:
            squared_distance_val: float
                squared euclidian distance between vector1 and vector2
        """
        # euclidian distance
        # squared_distance_val = np.linalg.norm(x1 - x2) ** 2
        squared_distance_val = np.dot((x1 - x2).T, (x1 - x2))
        return squared_distance_val
    

    def euclidian_distance(self, x1, x2):
        """
        Euclidian distance

        Args:
            x1: array-like
                vector1

            x2: array-like
                vector2
        
        Returns:
            distance_val: float
                distance between vector1 and vector2
        """
        # euclidian distance
        distance_val = np.linalg.norm(x1 - x2)
        # distance_val = np.sqrt(np.dot((x1 - x2).T, (x1 - x2)))
        return distance_val
    

    def gaussian_rbf(self, x1, x2, beta_val=1):
        """
        Gaussian radial basis function

        Args:
            x1: array-like
                vector1

            x2: array-like
                vector2

            beta_val: float, optional(default=1)
        """
        # phi_val = np.exp(- beta_val * self.squared_distance_func(x1, x2))
        phi_val = np.exp(- beta_val * self.squared_norm_func(x1, x2))
        return phi_val
    

    def thin_plate_spline_rbf(self, x1, x2, beta_val=1):
        """
        TPS(thin plate spline) radial basis function

        Args:
            x1: array-like
                vector1

            x2: array-like
                vector2

            beta_val: float, optional(default=1)
                is not used
        """
        phi_val = self.squared_norm_func(x1, x2) * np.log(self.norm_func(x1, x2) + self._eps)
        return phi_val
    
    
    def rbf_matrix(self, train_x_arr, center_x_arr, beta_val=1):
        """
        centerにフィットさせる場合はtrain_x_arrにcenterと同じものを入れる

        Args:
            train_x_arr: array-like
                shape=(train_n_samples, train_n_dimension)

            center_x_arr: array-like
                shape=(center_n_samples, center_n_dimension)
        """
        train_n_samples = len(train_x_arr)
        center_n_samples = len(center_x_arr)

        # ndarrayに直す
        # pd.DataFrameの場合rowとindexが逆転する
        train_x_arr = np.array(train_x_arr)
        center_x_arr = np.array(center_x_arr)

        phi_arr = np.array([self.radial_basis_func(train_x_arr[ii], center_x_arr[jj], beta_val=beta_val) for ii in range(train_n_samples) for jj in range(center_n_samples)]).reshape((train_n_samples, center_n_samples))
        """
        # joblib並列処理
        phi_arr = np.array(
            Parallel(n_jobs=self.n_jobs, verbose=0)([
                delayed(self.radial_basis_func)(train_x_arr[ii], center_x_arr[jj], beta_val=beta_val) for ii in range(train_n_samples) for jj in range(center_n_samples)
                ])
            ).reshape((train_n_samples, center_n_samples))
        """
        return phi_arr


    def rbf_linear_combination(self, w_arr, train_x_arr, center_x_arr, beta_val=1):
        """
        linear combination of radial basis function

        Args:
            w_arr: array-like
                weights

            train_x_arr: array-like

            center_x_arr: array-like

            beta_val: float, optional(default=1)
                coefficient of gaussian.
        """
        return np.dot(self.rbf_matrix(train_x_arr, center_x_arr, beta_val=beta_val), w_arr)


    def rbf_loss_func(self, w_arr, train_x_arr, center_x_arr, train_y_arr, beta_val=1, lambda_val=1):
        """
        rbf loss function

        Args:
            w_arr: array-like
                weights
            
            train_x_arr: array-like
                data to pred
            
            center_x_arr: array-like
                data for pred
            
            train_y_arr: array-like
                data to pred
        
        Returns:
            loss_val: float
                value of loss function
        """
        phi_mat = self.rbf_linear_combination(w_arr, train_x_arr, center_x_arr, beta_val=beta_val)

        # variance
        train_y_var = np.dot(
            (train_y_arr - phi_mat).T,
            (train_y_arr - phi_mat)
        )

        loss_val = train_y_var + lambda_val * np.dot(w_arr.T, w_arr)

        return loss_val


    def train_rbf_loss_func(self, w_arr, train_x_arr, center_x_arr, train_y_arr, beta_val=1, lambda_val=1, n_divide=1):
        """
        train rbf model weights
        rbf_loss_func() / n_divide

        Args:
            w_arr: ndarray

            train_x_arr: ndarray
                features of train data

            center_x_arr: ndarray
                基底の中心

            train_y_arr: ndarray
                target values of train data

            beta_val: float

            lambda_val: float

            n_divide: float, optional(default=1)
        """
        loss_val = self.rbf_loss_func(w_arr, train_x_arr, center_x_arr, train_y_arr, beta_val, lambda_val) / n_divide
        print('\rloss: ' + str(loss_val), end=' ')
        return loss_val
    
    
    def rbf_loss_fprime(self, w_arr, train_x_arr, center_x_arr, train_y_arr, beta_val=1, lambda_val=1, n_divide=1):
        """
        differential of rbf loss function by weights

        Args:
            w_arr: ndarray

            train_x_arr: ndarray

            center_x_arr: ndarray
                基底の中心

            train_y_arr: ndarray

            beta_val: float

            lambda_val: float

            n_divide: float, optional(default=1)

        Returns:
            loss_fprime_val: float
        """    
        # ndarrayに直す
        # pd.DataFrameの場合rowとindexが逆転する
        train_x_arr = np.array(train_x_arr)

        train_n_sample = len(train_x_arr)

        # rbf行列
        phi_arr = self.rbf_matrix(train_x_arr, center_x_arr, beta_val=beta_val)
        
        # 数値計算の闇
        # loss_fprime_val = -2 * np.dot(train_y_arr.T, phi_arr) + np.dot( (np.dot(phi_arr.T, phi_arr) + 2 * lambda_val * np.identity(train_n_sample)), w_arr)
        loss_fprime_val = -2 * np.dot(train_y_arr.T, phi_arr) + np.dot(np.dot(phi_arr.T, phi_arr), w_arr) + 2 * lambda_val * w_arr
        loss_fprime_val /= n_divide
        return loss_fprime_val
    
    
    def fit(
        self,
        data_x,
        data_y,
        w0_arr=None,
        beta_val=None,
        lambda_val=None,
        n_divide=None,
        n_samples=None,
        rel_ct=None,
        sampling_method=None,
        random_state=57
        ):
        """
        RBFでフィッティングを行いモデルを作成する.
        
        Args:
            data_x: array-like
                features
            
            data_y: array-like
                target variable
            
            sampling_method: str, optional(default='random')
        """
        
        # 使用するデータをndarrayに変換
        x_arr = np.array(data_x)
        y_arr = np.array(data_y)

        # 全体の行数と次元数
        n_rows = x_arr.shape[0]
        n_dim = x_arr.T.shape[0]

        # xとyを結合
        xy_arr = np.vstack((x_arr.T, y_arr.T)).T
        
        # n_samples更新有無確認のため値を保存
        old_n_samples = self.params['n_samples']

        if (not n_samples is None)and(type(n_samples)==int):
            # self.n_samples = n_samples
            self.params['n_samples'] = n_samples
        
        if (not beta_val is None)and((type(beta_val)==float)or(type(beta_val)==int)):
            # self.beta_val = beta_val
            self.params['beta_val'] = beta_val
        
        if (not lambda_val is None)and((type(lambda_val)==float)or(type(lambda_val)==int)):
            # self.beta_val = lambda_val
            self.params['lambda_val'] = lambda_val
        
        if (not n_divide is None)and((type(n_divide)==float)or(type(n_divide)==int)):
            # self.n_divide = n_divide
            self.params['n_divide'] = n_divide
        
        if self.params['n_samples'] >= 1:
            self.params['n_samples'] = int(min([self.params['n_samples'], n_rows]))
        else:
            self.params['n_samples'] = int(n_rows)
        
        # n_samplesが変更になる場合は重み初期値を変更しておく
        if self.params['n_samples']!=old_n_samples:
            self.params['w0_arr'] = np.zeros(shape=self.params['n_samples'])
        
        if (not rel_ct is None)and(type(rel_ct)==str):
            self.params['rel_ct'] = rel_ct
        
        if (not sampling_method is None)and(type(sampling_method)==str):
            self.params['sampling_method'] = sampling_method
        

        # center pointsを取得する必要がある場合する
        if not self.center_is_set:
            
            # random seedの設定
            np.random.seed(random_state)
            
            # samplingを行い, xy of center pointsを作成する
            if (self.params['sampling_method'] is None):

                print('error: center points are not set.')
                return False
            
            elif self.params['sampling_method']=='random':

                center_idx_arr = np.random.choice(np.arange(n_rows), size=self.params['n_samples'], replace=False)
                center_xy_arr = xy_arr[center_idx_arr]
                center_x_arr = center_xy_arr.T[:n_dim].T
                center_y_arr = center_xy_arr.T[n_dim].T
                self.set_center_points(center_x_arr, center_y_arr)
                self.center_idx_arr = center_idx_arr
                self.center_is_sampled = True

            elif self.params['sampling_method']=='last':

                center_idx_arr = np.arange((n_rows-self.params['n_samples']), n_rows)
                center_xy_arr = xy_arr[(n_rows-self.params['n_samples']):]
                center_x_arr = center_xy_arr.T[:n_dim].T
                center_y_arr = center_xy_arr.T[n_dim].T
                self.set_center_points(center_x_arr, center_y_arr)
                self.center_idx_arr = center_idx_arr
                self.center_is_sampled = True

            else:
                print('warning: sampling_method is not illegal.')
                center_idx_arr = np.random.choice(np.arange(n_rows), size=self.params['n_samples'], replace=False)
                center_xy_arr = xy_arr[center_idx_arr]
                center_x_arr = center_xy_arr.T[:n_dim].T
                center_y_arr = center_xy_arr.T[n_dim].T
                self.set_center_points(center_x_arr, center_y_arr)
                self.center_idx_arr = center_idx_arr
                self.center_is_sampled = True

                # return False
                self.params['sampling_method'] = 'random'
        

        # train dataの作成
        if self.params['rel_ct']=='equal':

            if self.center_is_sampled:
                # train_xy_arr = center_xy_arr
                train_x_arr = self.center_x_arr
                train_y_arr = self.center_y_arr
            
            else:
                print('error: center is not sampled')

        # center pointsとの重複を許す
        elif self.params['rel_ct']=='all':
            train_xy_arr = xy_arr

            train_x_arr = train_xy_arr.T[:n_dim].T
            train_y_arr = train_xy_arr.T[n_dim].T
        
        # center pointsとの重複を許さない
        elif self.params['rel_ct']=='exclusive':

            if self.center_is_sampled:
                train_idx_arr = np.setdiff1d(np.arange(n_rows), self.center_idx_arr)
                train_xy_arr = xy_arr[train_idx_arr]

                train_x_arr = train_xy_arr.T[:n_dim].T
                train_y_arr = train_xy_arr.T[n_dim].T
            
            else:
                print('error: center is not sampled')
        
        else:
            print('warning: rel_ct is not illegal.')

            if self.center_is_sampled:
                # train_xy_arr = center_xy_arr
                train_x_arr = self.center_x_arr
                train_y_arr = self.center_y_arr

                self.params['rel_ct'] = 'equal'
            
            else:
                print('error: center is not sampled')


        # 重みの初期化
        if (not w0_arr is None):
            self.params['w0_arr'] = w0_arr
        
        elif self.params['w0_arr'] is None:
            # self.params['w0_arr'] = np.ones(shape=self.params['n_samples'])
            self.params['w0_arr'] = np.zeros(shape=self.params['n_samples'])
        
        # print debug text
        if self.params['print_debug']:
            debug_text = '***** ApproximationRBFInterpolation debug mode *****\n'
            debug_text += 'distance: ' + str(self.params['distance']) + '\n'
            debug_text += 'RBF: ' + str(self.params['rbf']) + '\n'
            debug_text += 'data_x.shape: ' + str(x_arr.shape) + '\n'
            debug_text += 'data_y.shape: ' + str(y_arr.shape) + '\n'
            debug_text += 'train_x.shape: ' + str(train_x_arr.shape) + '\n'
            debug_text += 'train_y.shape: ' + str(train_y_arr.shape) + '\n'
            debug_text += 'center_x.shape: ' + str(center_x_arr.shape) + '\n'
            debug_text += 'center_y.shape: ' + str(center_y_arr.shape) + '\n'
            debug_text += 'n_rows: ' + str(n_rows) + '\n'
            debug_text += 'n_dims: ' + str(n_dim) + '\n'
            debug_text += 'n_samples: ' + str(self.params['n_samples']) + '\n'
            debug_text += 'sampling_method: ' + str(self.params['sampling_method']) + '\n'
            debug_text += 'rel_ct: ' + str(self.params['rel_ct']) + '\n'
            debug_text += 'initial weights: ' + str(self.params['w0_arr']) + '\n'
            debug_text += 'beta value: ' + str(self.params['beta_val']) + '\n'
            debug_text += 'lambda value: ' + str(self.params['lambda_val']) + '\n'
            debug_text += 'gtol: ' + str(self.params['gtol']) + '\n'
            debug_text += 'opt_disp: ' + str(self.params['opt_disp']) + '\n\n'
            
            self.debug_log += debug_text
            print(debug_text)
        

        # optimization for fitting
        self.rbf_wopt = fmin_bfgs(
            f=self.train_rbf_loss_func,
            x0=self.params['w0_arr'],
            fprime=self.rbf_loss_fprime,
            args=(train_x_arr, self.center_x_arr, train_y_arr, self.params['beta_val'], self.params['lambda_val'], self.params['n_divide']),
            gtol=self.params['gtol'],
            disp=self.params['opt_disp']
        )
        
        self.is_trained = True

        if self.params['print_debug']:
            best_loss_val = self.train_rbf_loss_func(self.rbf_wopt, train_x_arr, self.center_x_arr, train_y_arr, self.params['beta_val'], self.params['lambda_val'], self.params['n_divide'])
            debug_text += 'loss_function_value: ' + str(best_loss_val) + '\n\n'
            print('\n\n**************************************************\n\n')
        
        #print('RBF model is trained!')
        return self
    
    
    def predict(self, data_x):
        """
        予測を行う.

        Args:
            data_x: array-like
                features
        
        Returns:
            pred_y_arr: array-like
                prediction value
        """
        if not self.is_trained:
            print('error: model is not trained!')
            return False
            
        x_arr = np.array(data_x)
        pred_y_arr = self.rbf_linear_combination(w_arr=self.rbf_wopt, train_x_arr=x_arr, center_x_arr=self.center_x_arr, beta_val=self.params['beta_val'])
        return pred_y_arr
    
    
    def score(self, data_x, data_y, criterion='r2'):
        """
        return evaluation score

        Args:
            data_x: array-like
                features

            data_y: array-like
                target
            
            criterion: str
                metrics
        
        Returns:
            score_value: float
                evaluation value
        """

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


