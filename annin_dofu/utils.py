#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: laplaciannin102(Kosuke Asada)
@date: 2021/10/13

便利modules.

@history:
    2021/10/13:
        初期版作成.
    
    2021/11/22:
        get_dir_tree追加.
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
    'make_empty_file',
    'get_dir_tree'
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


def _get_dir_tree_rescursive(
    dir_path,
    pre_txt='',
    indent='  ',
    name_prefix='─',
    linesep='auto',
    add_dir_slash=False,
    max_depth=-1,
    depth=0
):
    """
    get_dir_tree内でのみ使用する関数.
    通常呼び出されない.
    tree出力のため再帰的に呼び出される.
    """
    # 改行文字列
    if linesep == 'auto':
        linesep = os.linesep
    
    pre_txt = str(pre_txt)
    paths_list = sorted(list_segments(dir_path=dir_path, forced_slash=True))
    dir_tree_txt = ''

    for path in paths_list:
        
        # ディレクトリかどうか.
        is_dir = os.path.isdir(path)

        # 親ディレクトリの中で最後のパスかどうか.
        is_last = False

        if path == paths_list[-1]:
            is_last = True
        
        # file or directory name
        name = path.split('/')[-1]
        
        # ディレクトリの最後にスラッシュを付ける場合.
        if is_dir and add_dir_slash and (name[-1]!='/'):
            name += '/'
        
        # 最後のパス.
        if is_last:
            branch_txt = '└'
            child_parent_branch_txt = '{indent}'.format(
                indent=indent
            )
        # 最後のパスでないとき.
        else:
            branch_txt = '├'
            child_parent_branch_txt = '│'

        # 1行分の文字列
        line_txt = '{pre_txt}{branch_txt}{name_prefix}{name}{linesep}'.format(
            pre_txt=pre_txt,
            branch_txt=branch_txt,
            name_prefix=name_prefix,
            name=name,
            linesep=linesep
        )
        
        child_pre_txt = '{pre_txt}{child_parent_branch_txt}{indent}'.format(
            pre_txt=pre_txt,
            indent=indent,
            child_parent_branch_txt=child_parent_branch_txt
        )

        dir_tree_txt += line_txt

        # pathがディレクトリを指すものだった場合,
        # さらに下の階層についてもディレクトリ構造を調べる.
        if os.path.isdir(path):
            
            # 深さを1深くする.
            child_depth = depth + 1
            
            # 最下層まで探索する(-1) or 最深層まで達してない場合.
            if (max_depth == -1) or (child_depth <= max_depth):

                dir_tree_txt += _get_dir_tree_rescursive(
                    dir_path=path,
                    pre_txt=child_pre_txt,
                    indent=indent,
                    name_prefix=name_prefix,
                    linesep=linesep,
                    add_dir_slash=add_dir_slash,
                    max_depth=max_depth,
                    depth=child_depth
                )

    return dir_tree_txt


def get_dir_tree(
    dir_path='./',
    indent='  ',
    name_prefix='─',
    linesep='auto',
    add_dir_slash=False,
    max_depth=-1,
    print_tree=True
):
    """
    dir_path以下のディレクトリ構造を木の形式の文字列として返す.
    print_treeをTrueにした時, 標準出力する.
    
    Args:
        dir_path: str, optional(default='./')
            検索対象のディレクトリのパス.
        
        indent: str, optional(default='  ')
            インデントに使用する文字列.
        
        name_prefix: str, optional(default='─')
            ファイル名またはディレクトリ名の頭に付ける文字列.
        
        linesep: str, optional(default='auto')
            改行に使用する文字列.
            'auto': OSで使用される改行文字列を自動で取得する.
            '\\n': LF
            '\\r\\n': CRLF
        
        add_dir_slash: bool, optional(default=False)
            ディレクトリ名の最後にスラッシュを付けるかどうか.
        
        max_depth: int, optional(default=-1)
            検索対象とするディレクトリの最大の深さ.
            -1: 最深層まで検索する.
            0: 最も浅い層のみ検索する.
        
        print_tree: bool, optional(default=True)
            ディレクトリ構造の木の形式の文字列を標準出力するかどうか.
    
    Returns:
        dir_tree_txt: str
            dir_path以下のディレクトリ構造の木の形式の文字列.
    
    Examples:
        >> get_dir_tree('../')
    """
    dir_path = str(dir_path)
    indent = str(indent)
    name_prefix = str(name_prefix)
    linesep = str(linesep)
    
    # 改行文字列
    if linesep == 'auto':
        linesep = os.linesep
    
    tmp_dir_path = dir_path
    
    if add_dir_slash and (tmp_dir_path[-1]!='/'):
        tmp_dir_path += '/'
    
    dir_tree_txt = '{tmp_dir_path}{linesep}'.format(
        tmp_dir_path=tmp_dir_path,
        linesep=linesep
    )
    
    # ディレクトリ木構造の文字列を取得する.
    dir_tree_txt += _get_dir_tree_rescursive(
        dir_path=dir_path,
        indent=indent,
        name_prefix=name_prefix,
        linesep=linesep,
        add_dir_slash=add_dir_slash,
        max_depth=max_depth,
        depth=0
    )
    
    # ディレクトリ木構造を標準出力する場合.
    if print_tree:
        print(dir_tree_txt)

    return dir_tree_txt

