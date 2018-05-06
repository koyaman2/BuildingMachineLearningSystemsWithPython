# クラス分類：悪い回答を判別する

### この章の内容を一言でいうと：増え続けるテキストのクラス分類にはk近傍法は向かないぜ。イケてないと思ったらバイアスーバリアンスのトレードオフをしよう！
実際に分類器を作ってみて試行錯誤する流れになってます
### この章の流れ
- データを入手する
- データを前処理する
- k近傍法で分類器を作成する
- 仮説を色々立てて特徴量を模索して試す
- 何したいんだっけ？を再確認する
- 改善案を考える
- バイアスーバリアンスのトレードオフを考える
- モデルを変える（k近傍法→ロジスティック回帰)
- 適合率と再現率を計算して向いてるやり方を模索する
- 回帰係数を使って分類器をスリムにする

## データを入手する
[ここから入手](http://archive.org/details/stackexchange)
右ペインの7zfiles>stackoverflow.com-Posts.7z を選択してダウンロード
※めっちゃ重い

## データを前処理する
- データを2011年以降に限定する
- 必要な属性のみに選別する

|属性             |備考|
----|----
|PostTypeId      |文書タイプの分類のために残す|
|CreationDate    |質問が投稿されてから回答が投稿されるまでの時間が良い回答になるかもなので残す|
|Score           |コミュニティの評価。大事なので残す|
|ViewCount       |回答が投稿された時点で０なので不要|
|Body            |中身なので重要。HTMLの形式からテキストに変換する|
|OwnerUserId     |ユーザーに関連した特徴を今回は考えないので不要と判断|
|Title           |今回は不要|
|CommentCount    |回答が投稿された時点では役に立たないので不要|
|AcceptedAnswerId|受理されたかどうかになるのでIsAcceptedという新しい属性を追加する|

- 良い回答を定義する
    - 良い回答/悪い回答の二項ではなく
    - スコアが0より大きければ良い、0より小さければ悪い、にした方が本質的
    ※良い回答ばかりの場合、良い回答/悪い回答の二項だと良い回答が悪い方に分類されてしまう

★ここ、テスト技法である同値分割の考え方にそっくりで奥が深い。
何を持って有効とするか、無効とするか。

```
all_answers = [q for q, v in meta.iteriterms() if v[‘ParentId’]!=-1]
Y = np.asarray([meta[aid][‘Score’]>0 for aid in all_answers])
```

## k近傍法で分類器を作成する
sklearn.neighborsというツールキットを使う
```
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=2)
print(knn)
```
- fit()：訓練を行う
- predict()：新しいデータに対しラベルを推測する
```
train_data = [[1], [2], [3], [4], [5], [6]]
train_label = [0,0,0,1,1,1]
knn.fit(train_data, train_label)
knn.predict(1.5)
knn.predict(37)
knn.predict(3)
```
- predict_proba()：結果に対する確率を得る
```
knn.predict_proba(1.5)
knn.predict_proba(37)
knn.predict_proba(3.5)
```

## 仮説を色々立てて特徴量を模索して試す
どのような特徴量を分類器に入力すべきか？
どのような特徴量が最も識別性を持つか？

分類器には数値しか入力できないので、
Textという属性は使えない

仮説：文書の中に多くのURLリンクが存在すればするほど良い回答である可能性が高くなる
→リンクの数をカウントする
```
import re
# 正規表現を用いて[ソースコード]と[URLリンク]を見つける
code_match = re.compile(‘<pre>(.*?)</pre>’, re.MULTILINE | re.DOTALL)
link_match = re.compile(‘<a href=“http://.*?”.*?>(.*?)</a>’, re.MULYILINE | re.DOTALL)

def extract_features_from_body(s):
  link_count_in_code = 0
  # コード中に存在するリンクをカウントする
  for match_str in code_match.findall(s):
    link_count_in_code += len(link_match.findall(match_str))
  return len(link_match.fnidall(s)) - link_count_in_code
```
先ほど定義したラベルの配列であるYと、対応する特徴量の配列を一緒にしてkNN分類器に入力する
```
X = np.asarray([extract_features_from_body(text) for post_id,
  text in fetch_posts() if post_id in all_answers])
knn = neighbors.KNeighborsClassifier()
knn.fit(X,Y)
```

## 何したいんだっけ？を再確認する
「何を評価したいのか？」をあらためてはっきりさせなければならない

テストデータに対して正しく予測した割合：最も簡単な評価方法は以下。

- 1〜0の間でプロットされる
    - 全て正しく予測をした場合は1
    - 全て誤った予測をした場合は0
- knn.score()で求められる

交差検定をする＝sklearn.cross_validationのKFoldクラスを使う
交差検定の各試行での正解率を平均して最終的な正解率を計算する
```
from sklearn.cross_validation import KFold
scores =[]
cv = KFold(n=len(X), k=10, indices=True)

for train, test in cv:
  X_train, Y_train = X[train], Y[train]
  X_test, Y_test = X[test], Y[test]
  clf = neighbors.KNeighborsClassifier()
  clf.fit(X, Y)
  scores.append(clf.score(X_test, Y_test))

print(“Mean(scores)=%.5f¥tStddev(scores)=%.5f”%(np.mean(scores, np.std(scores)))
```
結果が49%なので利用できるレベルでない
リンクの数は良い指標ではないかも？


仮説２：文書中に含まれるソースコードの行数も良い文書を示す要素かもしれない。あとソースコード部分以外の単語の数も

```
def extract_features_from_body(s):
  num_code_lines = 0
  link_count_in_code = 0
  code_free_s = s

# ソースコードを取り除いて行数を数える
for match_str in code_match.findall(s):
  num_code_lines += match_str.count(‘¥n’)
  code_free_s = code_match.sub(“”, code_free_s)
  # ソースコードにはリンクが含まれることがあるので、その場合はカウントしない
  link_count_in_code += len(link_match.findall(match_str))

links = link_match.findall(s)
link_count = len(links)
link_count -= link_count_in_code
html_free_s = re.sub(“ +”, “ “, tag_match.sub(‘’, code_free_s)).replace(“¥n”,””)
link_free_s = html_free_s

# 単語の数をカウントする前にリンクを削除
for link in links:
  if link.lower().startswith(“http://“):
    link_free_s = link_free_s.replace(link, ‘’)

  num_text_tokens = html_free_s.count(“ “)

return num_text_tokens, num_code_lines, link_count
```
ソースコードの行数(NumCodeLines)よりも単語の数(NumTextTokens)の方が変化に富んでいる。
```
print(“Mean(scores)=%.5f¥tStddev(scores)=%.5f”%(np.mean(scores, np.std(scores)))
```
結果が58%。少しだけ改善したがまだまだ。

さらに特徴量を追加！ドン！
- 単語数の平均：AvgSentLen
- 文書数の各単語の文字数：AvgWordLen
- 全ての文字が大文字で書かれている単語の数：NumAllCaps
- 感嘆符（！）の数：NumExclams

結果が57%に下がった。
→5NNを用いているため、他の文書の中から距離が近い順に５つの文書を選ぶ
　→新しい文書が属するクラスは５つの近傍文書で最も多くを占めるクラス
文書間の距離はユークリッド距離で計算される
※分類器を初期化するときに距離パラメータを指定しなかったので、デフォルトp=2になる

|文書|NumLinks|NumTextTokens|
----|----|----
|A|2|20|
|B|0|25|
|New|1|23|

上記の条件の場合、リンクの数の方が単語数よりも重要なので
NewはAに似ているべきだが、今の手法だとBに似ていると判断される
→今扱っているデータに対してk近傍法ではうまく分類できない

## 改善案を考える
- データを追加する
- モデルの複雑さを調整する
- 特徴量を修正する
- モデルを変更する

だいたいこういうことやるけど、時間かかる。
情報に基づいた決定を行うにはバイアスーバリアンスのトレードオフをする

## バイアスーバリアンスのトレードオフを考える

- モデルが単純すぎる＝未学習＝データに対してバイアスが大きすぎる
- モデルが複雑すぎる＝過学習＝データに対してバリアンスが大きすぎる

理想は両方とも小さくしたいが、シーソーのような関係にある

### バイアスが大きい場合の対処法
- 訓練データをいくら追加しても、正解率は改善されない
- 特徴量の数を減らしても改善してされない

本質的な理由はモデルが単純だから。なので対処法としては以下になる。

- 特徴量を増やす
- モデルを複雑なものにする
- モデルを変更する

### バリアンスが大きい場合の対処法
- よりデータを集める
- モデルの複雑さを減らす
　→kを増やす、特徴量の数を減らす

### 今回の問題はバイアスが大きいのかバリアンスが大きいのか？
バリアンスが大きい
→データを集めるか、kを増やすか、特徴量の数を減らす

実際に実験。
- データを集める→効果なし
- 特徴量の数を減らす→効果なし
- kを増やす→良い結果になるが十分ではない。
    - k5=57%
    - k90=62.8%
- そもそも90個の近傍点を見つけるのにも時間がかかる
→つまりk近傍法を使うには向かない

今回のシナリオでは、時間が経過するにつれて投稿される文書の数が増え続ける。
k近傍法は事例に基づくinstance-based学習であるため、訓練データとして用いたデータを保存しておく必要がある
そのため文書の数が増えれば増えるほど、分類するために必要な時間が長くなる
これはモデルベースのアプローチ（データからモデルを作る）とは異なる

## モデルを変える（k近傍法→ロジスティック回帰)
### ロジスティック回帰
- 分類に関する手法
- テキストベースの分類問題において威力を発揮する
- ロジスティック関数で回帰を行い、その結果から分類を行う

### ロジスティック回帰の簡単な例
- 各データは特徴量と対応するクラス（ラベル）を持つ
- クラスは１か０を取る
- X軸＝特徴量
- Y軸＝対応するクラス

データにはノイズが含まれており、特徴料が１〜６の間では両方のクラスが存在している
そのため、０か１の離散的な値を直接出力するような関数をモデル化するよりは
特徴量Xがクラス１に属する確率をP(X)としてその確率関数をモデル化する方がいい
★この理由がいまいちピンとこないっす！

P(X)が0.5より大きければクラス１に。それ以外であればクラス０に分類する

ある関数についてその出力値がある決められた範囲に収まるようにモデル化することは数学的に難しい
★ここもなんでかわからんっす！
しかし確率関数を少し調整することで出力値が常に０か１の間に収まるようにすることはできる

オッズ比とその対数が必要
オッズ比：P/(1-P)

ある特徴量について
クラス１に属する確率が0.9の場合、つまり
　P(Y=1)=0.9
の場合、オッズ比はP(Y=1)/P(Y=0)=0.9/0.1=9になる。
つまり、この特徴量を持つデータは9:1の確率でクラス１に属する

P(Y=0.5)であれば、1:1になる

Pが大きくなるに従い、その対数の値も大きくなる
逆関数として考えると、マイナス無限大からぷらす無限大までの範囲を、０から１までの有限範囲に変換する関数になる

- 特徴量の線形な組み合わせについては次の線形方程式をもちいる
    - Yi = C0 + C1 Xi
- Yをlog(odds)に置き換える
    - log( Pi / 1-Pi ) = C0 + C1Xi
- Piについての解を求めると、以下になる
    - Pi = 1 / 1+e-(C0+C1Xi)

我々はデータセットのすべてのペアデータ（Xi,Pi)に対して、誤差が最小となるような係数（C0とC1)を求めること
scikit-learnを用いることでこの作業を簡単にできる
```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
print(clf)

clf.fit(X,Y)
print(np.exp(clf.intercept_), np.exp(clf.conf_.ravel()))

def lr_model(alf, X):
  return 1 / (1 + np.exp(-(clf.intercept_ + clf.coef_*X)))

print(“P(x=-1)=%.2f¥tP(x=7)=%.2f”%(lr_model(clf, -1), lr_model(clf, 7)))
```
### 今の問題にロジスティック回帰を適用
- 90NN : 62.8%
- LogReg C=1.00 : 62.9%
- LogReg C=0.1: 63.1%

90NNよりも少し良くなっているが、それほど良くなっていない
- C:ロジスティック回帰で正規化を行うためのパラメータ。
    - ※k近傍法におけるkと同じような役割

C=0.1の時のバイアスーバリアンスをみる
→今のモデルはバイアスが大きい
　→現在の特徴量に対するロジスティック回帰は未学習であり、データを正しくは捉えることができていない

良い分類器が作成できない原因
- 今回の課題において、データにノイズが含まれすぎている
- 設計した特徴量に問題がある＝設定した特徴量がクラスを正しく分類できるだけの能力を備えていない

## 適合率と再現率を計算して向いてるやり方を模索する
||分類器：陽性|分類器：陰性|
|:---:|:---:|:---:|
|事実：陽性|TP|FN|
|事実：陰性|FP|TN|

- TP:True Positive
- FN:False Negative
- FP:False Positive
- TN:True Negative

我々が求めていることは、ある回答を良い回答と予測するとき、もしくは悪いと予測するときのどちらか一方の場合で、その結果が高い確率で正しくなるようにしたい場合
- 適合率＝TP / TP + FP

我々が求めることができるだけ多くの良い回答または悪い回答を見つけるようにしたい場合
- 再現率＝TP / TP + FN

我々ができることは閾値を変更したときのTP/FP/FNの数を数えること
これから再現率と適合率の変移をプロットできる
```
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(Y_test, clf.predict(X_test))
```
片方のクラスを許容できる精度で分類できたとしても、もう片方のクラスを同じように許容できる制度で分類できるとは限らない

ので両方見てみる
→今回の場合は良い回答を予測する方が良さそう

再現率が４０％の時まで適合率が８０％を超える
閾値を求める
```
medium = np.argsort(scores)[len(scores)/2]
thresholds = np.hstack(([0], thresholds[medium]))
idx80 = precisions>=0.8
print(“P=%/2f R=%.2f thresh=%.2f” % ¥ (precision[idx80][0], recall[idx80][0], threshold[idx80][0]))
```
- 閾値を0.63に設定した場合、適合率は80%を超え、そのときの再現率は37%であることがわかる
- 実際に良い回答の1/3に対してだけ判定を下す
- しかし、予測結果の大部分は正しい結果

分類器が予測を行う時にこの閾値を適用するには、predict_proba()メソッドを使う

```
thresh80 = threshold[idx80][0]
probs_for_good = clf.predict_proba(answer_features)[:,1]
answer_class = probs_for_good>thresh80)

from sklearn.metrics import classification_report
print(classification_report(Y_test, clf.predict_proba [:,1]>0.63, target_names=[‘not accepted’, ‘accepted’]))
```

## 分類器をスリムにする
各特徴量が分類を行うためにどれだけ貢献しているか、について見識を得ることは大事。

ロジスティック回帰の場合、学習した結果である回帰係数(clf.coef_)を直接確認できる
この回帰係数は各特徴料がどれだけ影響力を持っているかを示す
★これ具体的にどうするのかがよくわかりませんでした。

- この係数が大きければ大きいほど、回答の良し悪しを分類する時にその特徴量が重要な役割を担っている
- マイナスの値を持つ回帰係数は、その値が小さければ小さいほど、ある回答を悪いと分類する時に置ける影響が大きくなると言える

### 分類を行うに際して
- LInkCount(NumLinks=リンクの個数)とNumExclams(=感嘆符の数)が最も大きな影響力を持つことがわかる
- 一方、 NumImagesとAvgSentLenはさほど重要ではない、NumImagesは基本的には無視されている
- 実際には画像が追記されてある回答がほとんどないため、意味をなさない。
    - つまり、この特徴量を用いなくとも、分類器の性能は変わらない
★結局のところ仮説として良いものでも、数が少ないと分類には向かない

最後にシリアライズする
```
import pickle
pickle.dump(clf, open(“logreg.dat”, “w”))
clf = pickle.load(open(“logreg.dat”, “r”))
```

# まとめ

増え続けるテキストのクラス分類にはk近傍法は向かないぜ。イケてないと思ったらバイアスーバリアンスのトレードオフをしよう！

- モデルが単純すぎる＝未学習＝データに対してバイアスが大きすぎる
- モデルが複雑すぎる＝過学習＝データに対してバリアンスが大きすぎる

### バイアスが大きい場合の対処法
- 特徴量を増やす
- モデルを複雑なものにする
- モデルを変更する

### バリアンスが大きい場合の対処法
- よりデータを集める
- モデルの複雑さを減らす
　→kを増やす、特徴量の数を減らす
