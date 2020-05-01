# Created by RitsukiShuto on 2020/05/01.
# sklearn iris
#
# ライブラリを追加
from sklearn.xsets import load_iris
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
