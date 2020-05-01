# Created by RitsukiShuto on 2020/05/01.
# sklearn iris
#
# ライブラリを追加
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential

import numpy as np
import matplotlib.pyplot as plt

# データセットを読み込み
x, t = load_iris(return_X_y=True)
print('x : ', x.shape)
print('t : ', t.shape)

# データ型を合わせる
x = x.astype('float32')
t = t.astype('int32')

# データセットを分割
x_train_val, x_test, t_train_val, t_test = train_test_split(
    x, t, test_size=0.3, random_state=0
)
x_train, x_val, t_train, t_val = train_test_split(
    x, t, test_size=0.3, random_state=0
)

# モデルを定義


class CNN(chainer.Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(),
            conv2=L.Convolution2D(),
            l1=L.Linear(),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)

        return self.l1(h)
