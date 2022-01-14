# ゼロから作る Deep Learning

[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch/images/deep-learning-from-scratch.png" width="200px">](https://www.oreilly.co.jp/books/9784873117584/)

## 2. パーセプトロン（Perceptron）

AND, OR, NAND, XOR

## 3. ニューラルネットワーク（NN : Neural Network）

### 3.1 パーセプトロンからニューラルネットワークへ

入力信号の総和を出力信号に変換する関数は、活性化関数（Activation Function）と呼ばれる。
活性化関数は入力信号の総和がどのように活性化するか（どのように発火するか）を決定する役割です。

活性化関数がパーセプトロンからニューラルネットワークへ進むための架け橋になる。

### 3.2 活性化関数

[ステップ関数のグラフ(ch03/step_function.py)](ch03/step_function.py)

[シグモイド関数のグラフ(ch03/sigmoid.py)](ch03/sigmoid.py)

[シグモイド関数とステップ関数の比較(ch03/sig_step_compare.py)](ch03/sig_step_compare.py)

ステップ関数とシグモイド関数はともに非線形関数である。
ニューラルネットワークの活性化関数に非線形関数を用いる必要がある。
なぜなら、線形関数を用いるとニューラルネットワークで層を深くする意味がなくなってしまう。

シグモイド関数はニューラルネットワークの歴史上、古くから利用されてきたが、
最近はReLU（Rectified Linear Unit）という関数が主に利用されている。

[ReLU関数(ch03/relu.py)](ch03/relu.py)

### 3.3 多次元配列の計算

Numpyの多次元配列を使った計算をマスターすれば、ニューラルネットワークの実装を効率的に進める。

### 3.4 3層ニューラルネットワークの実装

### 3.5 出力層の設計

ニューラルネットワークは、分類問題と回帰問題の両方に用いることができる。
分類問題か回帰問題かで、出力層の活性化関数を変更する必要がある。
一般的に回帰問題は恒等関数、分類問題はソフトマックス関数を使用する。

ソフトマックス関数の出力は、0から1.0の間の実数になる。
また、ソフトマックス関数の出力の総和は1になるため、
ソフトマックス関数の出力を「確率」と解釈することができる。

### 3.6 手書き数字認識

MNISTデータセットは手書き数字の画像セットで、0から9までの数字画像から構成される。
訓練画像が60,000枚、テスト画像が10,000枚用意されている。

[MNIST訓練画像の1枚の表示(ch03/mnist_show.py)](ch03/mnist_show.py)

[3層ニューラルネットワークの推論処理(ch03/neuralnet_mnist.py)](ch03/neuralnet_mnist.py)

[100枚の画像を1バッチとして纏めて推論処理(ch03/neuralnet_mnist_batch.py)](ch03/neuralnet_mnist_batch.py)

## 4. ニューラルネットワークの学習

### 4.1 データから学習する

データ駆動：機械学習はデータが命

特徴量と機械学習によるアプローチでは、人が特徴量を設計したが、
ニューラルネットワークは、画像に含まれる特徴量までも「機械」が学習する。

### 4.2 損失関数

ニューラルネットワークの学習で用いられる指標は、損失関数（Loss Function）と呼ばれる。

* 平均二乗誤差（MSE : Mean Square Error）
* 交差エントロピー誤差（Cross Entropy Error）

### 4.3 数値微分

[数値微分の例(ch04/gradient_1d.py)](ch04/gradient_1d.py)

複数の変数からなる関数の微分を偏微分という。

### 4.4 勾配

すべての変数の偏微分をベクトルとしてまとめたものが勾配（Gradient）という。

[勾配の例(ch04/gradient_2d.py)](ch04/gradient_2d.py)

勾配が示す方向は、各場所において関数の値が最も減らす方向である。

勾配方向へ進むことを繰り返すことで、損失関数の値を徐々に減らすのが勾配法（Gradient Method）である。

* 勾配降下法（Gradient Descent Method）
* 勾配上昇法（Gradient Ascent Method）

[勾配降下法の例(ch04/gradient_method.py)](ch04/gradient_method.py)

学習率（Learning Rate）のようなパラメータはハイパーパラメータという。
ニューラルネットワークのパラメータ（重みやバイアス）とは性質の異なるパラメータである。
ニューラルネットワークの重みやバイアスは訓練データと学習のアルゴリズムによって「自動」で獲得できるが、
学習率のようなハイパーパラメータは人の手によって設定される。
ハイパーパラメータをいろいろな値で試しながら、うまく学習できるケースを探す作業が必要となる。

[簡単なニューラルネットワークの勾配を求める(ch04/gradient_simplenet.py)](ch04/gradient_simplenet.py)

### 4.5 学習アルゴリズムの実装

ニューラルネットワークは、適応可能な重みとバイアスがあり、
この重みとバイアスを訓練データに適応するように調整することを「学習」と呼ぶ。
ニューラルネットワークの学習は次の4つの手順で行う。

1. ミニバッチ : 訓練データからランダムに一部のデータを選び出す。
2. 勾配の算出 : ミニバッチの損失関数を減らすために、各重みパラメータの勾配を求める。
3. パラメータの更新 : 重みパラメータを勾配方向に微小量だけ更新する。
4. 繰り返す : 1.~3.の手順を繰り返す。

使用するデータはミニバッチとして無作為に選ばれているため、
確率的勾配降下法（SGD : Stochastic Gradient Descent）と呼ばれる。

[2層ニューラルネットワークのクラス(ch04/two_layer_net.py)](ch04/two_layer_net.py)

[ミニバッチ学習の実装(ch04/train_neuralnet.py)](ch04/train_neuralnet.py)

## 5. 誤差逆伝播法

誤差逆伝播法は、重みパラメータの勾配の計算を効率よく行う手法である。

### 5.1 計算グラフ

計算グラフは、計算の過程をグラフによって表したものである。

### 5.2 連鎖律

計算グラフの順伝播は、計算の結果を順方向に伝達した。
逆方向の伝播では「局所的な微分」を、順方向とは逆向きに伝達していく。
この「局所的な微分」を伝達する原理は連鎖律（Chain Rule）によるものである。

### 5.3 逆伝播

加算ノードと乗算ノードの逆伝播の例をこの章で示す。

### 5.4 単純なレイヤの実装

[乗算レイヤ/加算レイヤの実装例(ch05/layer_naive.py)](ch05/layer_naive.py)

[乗算レイヤを使用した実装例(ch05/buy_apple.py)](ch05/buy_apple.py)

### 5.5 活性化関数レイヤの実装

計算グラフの考え方で、ニューラルネットワークを構成する「層（レイヤ）」を実装する。

[ReLUレイヤの実装(common/layers.py)](common/layers.py)

[Sigmoidレイヤの実装(common/layers.py)](common/layers.py)

### 5.6 Affine/Softmaxレイヤの実装

[Affineレイヤの実装(common/layers.py)](common/layers.py)

[Softmax-with-Lossレイヤの実装(common/layers.py)](common/layers.py)

### 5.7 誤差逆伝播法の実装

[誤差逆伝播法の実装](ch05/two_layer_net.py)

## 6. 学習に関するテクニック

### 6.1 パラメータの更新

ニューラルネットワークの学習の目的は、損失関数の値をできるだけ小さくするパラメータを見つけることである。
これは最適なパラメータを見つける問題で、このような問題を解くことを最適化（Optimization）という。

[確率的勾配降下法（SGD : Stochastic Gradient Descent）(common/optimizer.py)](common/optimizer.py)

[Momentum(common/optimizer.py)](common/optimizer.py)

[AdaGrad(common/optimizer.py)](common/optimizer.py)

[Adam(common/optimizer.py)](common/optimizer.py)

[最適化手法の比較(ch06/optimizer_compare_naive.py)](ch06/optimizer_compare_naive.py)

[MNISTデータセットによる更新手法の比較(ch06/optimizer_compare_mnist.py)](ch06/optimizer_compare_mnist.py)

### 6.2 重みの初期値

重みの初期値としてどのような値を設定するかで、ニューラルネットワークの学習の成否が分かれることがよくあります。

[隠れ層のアクティベーション分布(ch06/weight_init_activation_histogram.py)](ch06/weight_init_activation_histogram.py)

## 7. 畳み込みニューラルネットワーク

## 8. Deep Learning