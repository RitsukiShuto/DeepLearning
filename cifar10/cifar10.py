# Created by RitsukiShuto on 2020/04/19.
# Cifar10
# ライブラリを読み込み
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential

import numpy as np
import matplotlib.pyplot as plt

# データセットを読み込み


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# データの読み込みと加工


def load_cifar10():
    train_data = np.empty((0, 32*32*3))
    train_label = np.empty(1)

    for num_of_batch in range(1, 6):
        # データバッチを読み込み
        dict = unpickle(
            "cifar-10-batches-py/data_batch_{}".format(num_of_batch))
        # 1~6まで全て結合
        if num_of_batch == 1:
            train_data = dict[b'data']
            train_label = dict[b'labels']
        else:
            train_data = np.vstack((train_data, dict[b'data']))
            train_label = np.hstack((train_label, dict[b'labels']))

    # テストデータをロード
    dict = unpickle("cifar-10-batches-py/test_batch")
    test_data = dict[b"data"]
    test_label = dict[b"labels"]

    # データ型を変換
    train_data = np.array(train_data, dtype='float32')
    train_label = np.array(train_label, dtype='int32')
    test_data = np.array(test_data, dtype='float32')
    test_label = np.array(test_label, dtype='int32')

    # 画像のピクセルを正規化
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    return train_data, train_label, test_data, test_label

# ニューラルネットワークを定義して学習を行う


def train_cifar10(train_data, train_label):
    l = L.Linear(3, 2)  # ネットワークを定義

    net = Sequential(
        L.Linear(n_input, n_hidden), F.sigmoid,
        L.Linear(n_hidden, n_hidden), F.relu,
        L.Linear(n_hidden, n_output)
    )

    # 目的関数を定義
    lr = 0.075  # TODO: 調整 lr
    optimizer = chainer.optimizers.SGD(lr)
    print(optimizer.setup(net))

    # ニューラルネットワークを訓練
    n_epoch = 100        # TODO: 調整 epoch
    n_batchsize = 1024      # TODO: 調整 batchsize
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
        order = np.random.permutation(range(len(train_data)))  # データをランダムに並べ替え

        # 各バッチの目的関数と精度を保存
        loss_list = []
        accuracy_list = []

        for i in range(0, len(order), n_batchsize):
            index = order[i: i + n_batchsize]
            train_data_batch = train_data[index, :]
            train_label_batch = train_label[index]

            # 予測値を出力
            train_predicted_val = net(train_data_batch)

            # 目的関数から分類精度を計算
            loss_train_batch = F.softmax_cross_entropy(
                train_predicted_val, train_label_batch
            )
            accuracy_train_batch = F.accuracy(
                train_predicted_val, train_label_batch
            )

            loss_list.append(loss_train_batch.array)
            accuracy_list.append(accuracy_train_batch.array)

            # 勾配のリセット
            net.cleargrads()
            loss_train_batch.backward()

            optimizer.update()  # パラメータを更新

            iteration += 1  # カウントアップ

        # 訓練データに対する目的関数の出力と分類精度を集計 => 平均値
        loss_train = np.mean(loss_list)
        acuracy_train = np.mean(accuracy_list)

        # 訓練結果を出力
        print('epoch: {}, iteration: {}, loss(train): {:.4}, accuracy: {:.4}'.format(
            epoch, iteration, loss_train, acuracy_train
        ))

        # ログを保存
        results_train['loss'].append(loss_train)
        results_train['accuracy'].append(acuracy_train)

    chainer.serializers.save_npz('my_cifar10.net', net)  # 訓練済みネットワークを保存

    # 目的関数を出力
    plt.plot(results_train['loss'], label='train')
    plt.savefig('plt/loss-e{}lr{}bt{}H{}.png'.format(n_epoch,
                                                     lr * 1000, n_batchsize, n_hidden))
    plt.show()

    # 分類精度を出力
    plt.plot(results_train['accuracy'], label='accuracy')
    plt.savefig('plt/accuracy-e{}lr{}bt{}H{}.png'.format(n_epoch,
                                                         lr * 1000, n_batchsize, n_hidden))
    plt.show()

    return test_data, test_label

# 学習済みネットワークとテストバッチを用いて画像認識を行う


def inferrene_cifar10(test_data, test_label):
    # ニューラルネットを使って推論
    loaded_net = Sequential(
        L.Linear(n_input, n_hidden), F.sigmoid,
        L.Linear(n_hidden, n_hidden), F.relu,
        L.Linear(n_hidden, n_output)
    )

    chainer.serializers.load_npz('my_cifar10.net', loaded_net)

    with chainer.using_config('train', False), chainer.using_config('enablebackprop', False):
        results_test = loaded_net(test_data)

    print(np.argmax(results_test[0, :].array))

    return 0


# ネットワークの構造を定義
n_input = 3072
n_hidden = 2000  # TODO: 調整 hidden
n_output = 10

# データセットを読み込み
train_data, train_label, test_data, test_label = load_cifar10()

select_train = int(input("訓練を行いますか(Yes=>0/No=>1): "))

if select_train == 0:
    print("///////CIFAR-10の訓練を開始します.///////")
    # 訓練
    train_data, train_label = train_cifar10(train_data, train_label)

# 検証
inferrene_cifar10(test_data, test_label)
