import sys
import os

import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定


def get_data():
    """
    MNISTデータセットからテスト画像とテストラベルを取得する。
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """
    sample_weightに保存された学習済みの重みパラメータを読み込む。
    """
    with open('ch03/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 第1層
    # 入力層（784個ニューロン（画像サイズ28×28）） -> 隠れ層1（50個ニューロン）
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # 第2層
    ## 隠れ層1（50個ニューロン） -> 隠れ層2（100個ニューロン）
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # 第3層
    # 隠れ層2（100個ニューロン） -> 出力層（10個ニューロン（数字0から9の10クラス））
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
