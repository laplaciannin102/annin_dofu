#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

便利modules.

@history:
    2021/10/13:
        初期版作成.
"""



# --------------------------------------------------------------------------------
# Load modules
# --------------------------------------------------------------------------------

import sys, os
import gc
import time
from glob import glob
from pathlib import Path

# original modules
from .calc import *
# from .stats import *



# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

# all modules to import
__all__ = [
    'get_elapsed_time',
    'arrange_path_parser',
    'list_segments',
    'pro_makedirs',
    'make_empty_file'
]



# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------


def get_elapsed_time(start, print_time=True):
    """
    経過時間を取得する.
    使用前にtime.time()で処理開始時間を取得しておく.

    Args:
        start: float
            計測開始時間.
        
        print_time: bool, optional(default=True)
            経過時間を標準出力するかどうか.
    
    Returns:
        elapsed_time: float
            経過時間.
            単位: 秒(unit: seconds).
    """

    try:
        end = time.time()
        elapsed_time = float(end - start)
        rounded_elapsed_time = pro_round(num=elapsed_time, ndigits=1)

        if print_time:
            print('elapsed time:', rounded_elapsed_time, 's')
        
        return elapsed_time
    
    except Exception as e:
        print('error: get elapsed time')
        print(e)

        elapsed_time = -1

        return elapsed_time


def arrange_path_parser(path, is_directory=False):
    """
    pathで使用されるテキストのparserの前処理を行う.
    「backslash(円マーク)」を「slash」に変換する.
    ディレクトリの場合, 最後にslashが付いていなければ追加する.
    
    Args:
        path: str
            target path text
        
        is_directory: bool, optional(default=False)
            the path is path of directory
    
    Examples:
        print(arrange_path_parser('.\\sample_dir1\\sample_dir2\\sample.txt'))
        # ./sample_dir1/sample_dir2/sample.txt

        print(arrange_path_parser('./sample_dir1/sample_dir2/sample.txt'))
        # ./sample_dir1/sample_dir2/sample.txt

        print(arrange_path_parser('.\\sample_dir1\\sample_dir2', is_directory=True))
        # ./sample_dir1/sample_dir2/

        print(arrange_path_parser('.\\sample_dir1\\sample_dir2\\', is_directory=True))
        # ./sample_dir1/sample_dir2/
    """

    try:
        path_text = str(path)

        # 「backslash(円マーク)」を「slash」に変換する
        path_text = path_text.replace('\\', '/')
        
        # ディレクトリの場合, 最後にslashが付いていなければ追加する.
        if is_directory and (path_text[-1] != '/'):
            path_text += '/'
        
        return path_text
    
    except Exception as e:
        print('error:')
        print(e)

        path_text = path
        return path_text


def list_segments(
    dir_path='./',
    rescursive=False,
    only_fname=False,
    extension=None,
    forced_slash=False
):
    """
    dir_path以下にあるファイルの相対パスのリストを返す.

    Args:
        dir_path: str, optional(default='./')
            検索対象のディレクトリのパス.

        rescursive: bool, optional(default=False)
            再帰的に検索するかどうか.
            Trueの場合, より深い階層に存在するファイルのパスもリストに格納して返す.
        
        only_fname: bool, optional(default=False)
            ファイル名のみを取得するかどうか.
            Trueの場合, パスの全体ではなくファイル名のみをリストに格納して返す.

        extension: str, list of str or tuple of str, optional(default=None)
            検索対象とする拡張子. 'csv'や['csv', 'xlsx']の様な形で指定.
        
        forced_slash: bool, optional(default=False)
            parserとして「slash」を使用するかどうか.
            Trueの場合, パスのparserを強制的に「slash」に変換する.

    Returns:
        paths_list: list of str
            ファイルの相対パスのリスト.
    """
    # ディレクトリ
    dpath_obj = Path(dir_path)

    # 再帰的に検索するかどうか
    resc_path = './*'
    if rescursive:
        resc_path = '**/*'

    # 拡張子
    if extension is None:
        ext_list = ['']
    elif (type(extension) == tuple) or (type(extension) == list):
        extension = list(extension)
        ext_list = ['.' + str(ii) for ii in extension]
    else:
        ext_list = ['.' + str(extension)]

    # それぞれの拡張子について検索
    paths_list = []
    for ext in ext_list:
        paths_list += list(dpath_obj.glob(resc_path + ext))

    # strに直す
    paths_list = [str(ii) for ii in paths_list]
    # 重複の削除
    paths_list = sorted(set(paths_list), key=paths_list.index) # 3.6以降ではl=list(dict.fromkeys(l))でも

    # 強制的にslash記号を使用する場合
    if forced_slash:
        paths_list = [arrange_path_parser(ii, is_directory=False) for ii in paths_list]

    # ファイル名のみ取得する場合
    if only_fname:

        # ファイル名を格納するリストを宣言
        fnames_list = []

        for tmp_path in paths_list:
            # pathのparserを整える.
            tmp_path = arrange_path_parser(tmp_path, is_directory=False)
            # 最後の文字がslashであれば削除する.
            if tmp_path[-1]=='/':
                tmp_path = tmp_path[:-1]    
            # 最後のparser(slash)の右側にある文字列を取得する.
            fnames_list += [tmp_path.split('/')[-1]]
        
        # paths_listにfnames_listを代入する.
        paths_list = fnames_list

    return paths_list


def pro_makedirs(dir_path):
    """
    ディレクトリを作成する. 指定のディレクトリが存在しない場合のみ作成する.

    Arg:
        path: str
            ディレクトリのパス.
    """
    dir_path = str(dir_path)

    # ディレクトリが存在しない場合にのみディレクトリを作成する.
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        pass


def make_empty_file(dir_path='./', file_name='.gitkeep', encoding='utf-8'):
    """
    指定したディレクトリに空のテキストファイルを作成する.
    デフォルトでは.gitkeepファイルを作成する.
    
    Args:
        dir_path: str, optional(default='./')
            target directory path
        
        file_name: str, optional(default='.gitkeep')
            file name
        
        encoding: str, optional(default='utf-8')
            encoding of text file
    """
    # parserを整えてから
    # file pathを作成
    file_path = '{}{}'.format(
        arrange_path_parser(
            path=str(dir_path),
            is_directory=True
        ),
        str(file_name)
    )
    
    # file_pathに書き込む
    with open(file_path, mode='w', encoding=encoding) as f:
        f.write('')


