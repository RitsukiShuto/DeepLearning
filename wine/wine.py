# Created by RitsukiShuto on 2020/04/16.
# Sklearn boston
#
# import lib
import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# データの読み込み
x, t = load_wine(return_X_y=True)

print('X : ', x.shape)
print('t : ', t.shape)

# データ形式を合わせる
x = x.astype('float32')
t = t.astype('int32')

# データセットを分割
x_train_val, x_test, t_train_val, t_test = train_test_split(
    x, t, test_size=0.3, random_state=0)
x_train, x_val, t_train, t_val = train_test_split(
    x_train_val, t_train_val, test_size=0.3, random_state=0)

# ネットワークを定義
l = L.Linear(3, 2)

# netとしてインスタンス化
n_input = 13
n_hidden = 10
n_output = 3

net = Sequential(
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_output)
)

# 目的関数を定義
optimizer = chainer.optimizers.SGD(lr=0.01)
print(optimizer.setup(net))

# ニューラルネットを訓練
n_epoch = 30
n_batchsize = 16
iteration = 0

# ログ保存用
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}

for epoch in range(n_epoch):
    order = np.random.permutation(range(len(x_train)))

    # 各バッチ毎の目的関数と精度を保存
    loss_list = []
    acuracy_list = []

    # epoch / i
    for i in range(0, len(order), n_batchsize):
        # バッチを準備
        index = order[i: i + n_batchsize]
        x_train_batch = x_train[index, :]
        t_train_batch = t_train[index]

        # 予測値を出力
        y_train_batch = net(x_train_batch)

        # 目的関数から分類精度を計算
        loss_train_batch = F.softmax_cross_entropy(
            y_train_batch, t_train_batch)
        accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

        loss_list.append(loss_train_batch.array)
        acuracy_list.append(accuracy_train_batch.array)

        # 勾配のリセットと計算
        net.cleargrads()
        loss_train_batch.backward()

        # パラメータを更新
        optimizer.update()

        # カウントアップ
        iteration += 1

    # 訓練データに対する目的関数の出力と分類精度を集計 平均値
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(acuracy_list)

    # 1エポックごとに検証用データで評価
    # 検証用データで予測値を出力
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_val = net(x_val)

    # 目的関数から分類精度を計算
    loss_val = F.softmax_cross_entropy(y_val, t_val)
    accuracy_val = F.accuracy(y_val, t_val)

    # 結果を出力
    print('epoch: {}, iteration: {}, loss(train): {:.4f}, loss(valid): {:.4f}'.format(
        epoch, iteration, loss_train, loss_val.array))

    # ログを保存
    results_train['loss'].append(loss_train)
    results_train['accuracy'].append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    results_valid['accuracy'].append(accuracy_val.array)

# 目的関数の出力
plt.plot(results_train['loss'], label='train')
plt.plot(results_valid['loss'], label='valid')
plt.legend()
plt.show()

# 分類精度
plt.plot(results_train['accuracy'], label='train')
plt.plot(results_valid['accuracy'], label='valid')
plt.legend()
plt.show()

# テストデータで予測値を計算
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = net(x_test)

accuracy_test = F.accuracy(y_test, t_test)
print(accuracy_test.array)

chainer.serializers.save_npz('my_wine.net', net)

# ニューラルネットを使って推論
loaded_net = Sequential(
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_output)
)

chainer.serializers.load_npz('my_wine.net', loaded_net)

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = loaded_net(x_test)

print(np.argmax(y_test[0, :].array))
