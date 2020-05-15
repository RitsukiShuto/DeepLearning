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

# ログ保存用
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}

# 訓練
for epoch in range(n_epoch):
    order = np.random.permutation(range(len(x_train)))

    # 各バッチの目的関数と精度を保存
    loss_list = []
    accuracy_list = []

    # epoch / i
    for i in range(0, len(order), n_batchsize):
        # バッチを準備
        index = order[i: i + n_batchsize]       # BUG: 原因不明
        x_train_batch = x_train[index, :]
        t_train_batch = t_train[index]

        # 予測値を出力
        y_train_batch = net(x_train_batch)

        # 目的関数から分類精度を計算
        loss_train_batch = F.softmax_cross_entropy(
            y_train_batch, t_train_batch
        )
        accuracy_train_batch = F.accuracy(
            y_train_batch, t_train_batch
        )

        loss_list.append(loss_train_batch.array)
        accuracy_list.append(accuracy_train_batch.array)

        # 勾配のリセット
        net.cleargrads()
        loss_train_batch.backward()

        # パラメータを更新
        optimizer.update()

        # カウントアップ
        iteration += 1

    # 訓練データに対する目的関数の出力と分類精度を集計 => 平均値
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)

    # エポックごとに検証用データで評価
    with chainer.using_config('train', False), chainer.using_config('enablebackprop', False):
        y_val = net(x_val)

    # 目的関数から分類精度を計算
    loss_val = F.softmax_cross_entropy(y_val, t_val)
    accuracy_val = F.accuracy(y_val, t_val)

    # 結果を出力
    print('epoch: {}, iteration: {}, loss(train): {:.4}, loss(valid): {:.4f}'
          .format(epoch, iteration, loss_train, loss_val.array))

    # ログを保存
    results_train['loss'].append(loss_train)
    results_train['accuracy'].append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    results_valid['accuracy'].append(accuracy_val.array)
