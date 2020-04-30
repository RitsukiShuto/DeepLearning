# Created by RitsukiShuto on 2020/04/30.
# CNN_cifar10
#
# ライブラリを読み込み
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential

import pickle
import numpy as np
import matplotlib.pyplot as plt

# データセットを読み込み


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# データセットの読み込み


def load_cifar10():
    train_data = np.empty((0, 32 * 32 * 3))
    train_label = np.empty(1)

    for num_of_batch in range(1, 6):
        # データバッチを読み込み
        dict = unpickle(
            "cifar-10-batches-py/data_batch_{}".format(num_of_batch)
        )
        # 1 ~ 6まで結合
        if num_of_batch == 1:
            train_data = dict[b'data']
            train_label = dict[b'labels']
        else:
            train_data = np.vstack((train_data, dict[b'data']))
            train_label = np.hstack((train_label, dict[b'labels']))

        # テストデータをロード
        dict = unpickle("cifar-10-batches-py/test_batch")
        test_data = dict[b'data']
        test_label = dict[b'labels']

        # データ型を変換
        train_data = np.array(train_data, dtype='float32')
        train_label = np.array(train_label, dtype='int32')
        test_data = np.array(test_data, dtype='float32')
        test_label = np.array(test_label, dtype='int32')

        # 画像のピクセルを正規化
        train_data = train_data / 255.0
        test_data = train_data / 255.0

        return train_data, train_label, test_data, test_label
