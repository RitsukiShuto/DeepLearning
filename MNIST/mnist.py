# Created by RitsukiShuto on 2020/05/04.
#
# ライブラリを追加
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential

from chainer.datasets import mnist
from chainer.datasets import split_dataset_random

import numpy as np
import matplotlib.pyplot as plt

# データセットをロード
train_val, test = mnist.get_mnist(withlabel=True, ndim=1)

# データを例示
x_train, t_train = train_val[0]
plt.imshow(x_train.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
print('label = ', t_train)

# 訓練用(Training)と検証用(Validation)にデータセットを分割
train, valid = split_dataset_random(train_val, 50000, seed=0)

print('Training dataset size = ', len(train))
print('Validation dataset size = ', len(valid))

# 検証用
print(train)

# ハイパーパラメータを定義
n_epoch = 30
n_batchsize = 128
iteration = 0

# ニューラルネットを定義
n_input = 784
n_hidden = 1000
n_output = 10

net = Sequential(
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_output)
)

# 目的関数を定義
optimizer = chainer.optimizers.SGD(lr=0.01)
print(optimizer.setup(net))
