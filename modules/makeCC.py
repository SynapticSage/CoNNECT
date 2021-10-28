#
# makeCC.py
# Copyright (c) 2020 Daisuke Endo
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
#
#----------------------------------
# 作成者：Daisuke Endo
# 連絡先:daisuke.endo96@gmail.com
# 最終更新日　2020/5/17
#----------------------------------
# ここには2つの関数を記述している。
# 1. あるペアのCrossCorrelogramを計算する関数
# 2. 全てのペアのCrossCorrelogramを計算する関数
#----------------------------------

import cython
import numpy as np
import math
import sys
from joblib import Parallel, delayed
#from crosscorrelogram import computeCC, computeAutoCC
from crosscorrelogram_gpu import computeAutoCC, computeCC

max_block_len = 65536/2
def return_threads_and_blocks(spiketrain, i, j,
                              threads_per_block=64,
                              block_dim=2):
    Ni = len(spiketrain[i])
    Nj = len(spiketrain[j])
    spike_product = Ni * Nj
    spike_product /= threads_per_block
    sq_Nspike = spike_product**(-block_dim)
    chunks = (sq_Nspike // max_block_len) + 1
    N = np.ceil(np.min((sq_Nspike, max_block_len)))
    N = np.max((2, N)).astype('int')
    if block_dim == 1:
        blocks_per_grid = (N, 1, 1)
        threads_per_block = (threads_per_block, 1, 1)
    elif block_dim == 2:
        blocks_per_grid = (N, N, 1)
        threads_per_block = (threads_per_block, 1, 1)
    elif block_dim == 3:
        blocks_per_grid = (N, N, N)
        threads_per_block = (threads_per_block, 1, 1)

    return blocks_per_grid, threads_per_block, int(chunks)


def computeHist(i, j, spiketrain, Begin, End):
    """
    jから見た時のiの相対時刻を求めて、CrossCorrelogramを計算する。
    """
    spike0 = spiketrain[i]
    spike1 = spiketrain[j]

    #histogram = computeCC(spike1, spike0, Begin, End)
    histogram = np.zeros(shape=int(End-Begin),
                         dtype=np.int64)
    blocks_per_grid, threads_per_block, chunks = return_threads_and_blocks(spiketrain, i, j)
    if chunks > 1:
        for chunk in range(chunks):
            start, end = 0 + chunk*max_block_len, \
                         np.max((chunk*max_block_len, len(spike1)))
            computeCC[blocks_per_grid, threads_per_block](histogram, spike1[start:end], spike0, Begin, End)
    else:
        computeCC[blocks_per_grid, threads_per_block](histogram, spike1, spike0, Begin, End)

    return i, j, histogram

def makeCC_allpair(spikes, Begin, End, N_thred):
    """
    全てのCCをBegin~Endの間で計算する
    args:
        spikes: list型
        Begin, End: int型
        N_thred: 並列計算をするときに与えるスレッド数
    return:
        X: ペアごとに計算したCC. X.shape = (ニューロンのペア数、CCの幅)
        index: ペアのニューロンの番号. index.shape = (ニューロンのペア数、2). index[i][0]は結合先ニューロン、index[i][1]は結合元のニューロンを表す.
    """
    #-----------------------------------------------------
    def func(i, j, spiketrain, Begin, End):
        """
        並列計算の中に入れる関数.
        Auto CrossCorrelogramを計算する場合と別のニューロン間のCrossCorrelogramで分けている.
        """
        if i > j:
            pass
            # print("Because {} > {}, we pass this pair".format(i, j))
        elif i == j:
            # print("Because {} == {}, we compute autoCC".format(i, j))
            histogram = np.zeros(shape=int(End-Begin),
                                 dtype=np.int64)
            blocks_per_grid, \
                threads_per_block, chunks = return_threads_and_blocks(spiketrain, i, i)
            #histogram = computeAutoCC(spiketrain[i], Begin, End)
            if chunks <= 1:
                computeAutoCC[blocks_per_grid, threads_per_block](histogram, spiketrain[i], Begin, End)
            else:
                for chunk in range(chunks):
                    start, end = 0 + chunk*max_block_len, \
                                 np.max((chunk*max_block_len, len(spiketrain[i])))
                    computeAutoCC[blocks_per_grid, threads_per_block](histogram, spiketrain[i], Begin, End)
            return i, j, histogram
        else:
            return computeHist(i, j, spiketrain, Begin, End)
    #-----------------------------------------------------
    # 並列計算でCCを計算する
    N_neuron = len(spikes)
    print("Compute Cross-Correlogram between {} ms and {} ms".format(Begin, End))

    if N_thred > 1:
        with tqdm_joblib(tqdm(desc="computing CC", total=N_neuron**2)) as progress_bar:
            result = Parallel(n_jobs=N_thred)(
                [delayed(func)(i, j, spikes, Begin, End)
                 for i in range(N_neuron)
                 for j in range(N_neuron)])
    else:
        result = []
        from itertools import product
        for i, j in tqdm(product(range(N_neuron), range(N_neuron)), desc="computing CC", total=N_neuron**2):
            result.append(func(i, j, spikes, Begin, End))

    # 計算結果を扱いやすい形に変更する
    X = []
    index = []
    for _ in range(len(result)):
        # 計算していないデータ(i<j)はNoneが入っているのでpassする
        if result[_] == None:
            pass
        else:
            # i, jのこと. neuronのindexに変換する
            post = result[_][0]
            pre = result[_][1]
            cc = result[_][2]

            if post == pre:
                # i == jについて
                index.append([post, pre])
                X.append(cc)

            else:
                # i < jについて
                index.append([post, pre])
                X.append(cc)

                # i > j. 反転した方向も作る
                index.append([pre, post])
                X.append(cc[::-1])

    return np.array(X), np.array(index)

import contextlib
import joblib
from tqdm import tqdm    
from joblib import Parallel, delayed
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
