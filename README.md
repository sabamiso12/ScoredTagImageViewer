# ScoredTagImageViewer
![image](https://user-images.githubusercontent.com/128471860/226951063-24f5b25d-4492-460f-bafa-2f09e3ff8029.png)

## これなに？
スコア付きタグ情報のある画像を閲覧したりスコアで並べ替えたりフィルタリングしたりできるツールです。

[StableDiffusion](https://github.com/CompVis/stable-diffusion)を追加学習する際の画像の選別やタグ付けのお供にどうぞ。

下記のような要望に応えられると思います。
* 「artist_name」は低い閾値でタグ付けしたい。
* 「multiple_views」の要素が少しでもある画像は除外したい。
* 「medium/large_XXXX」は点数高い方のタグだけ残したい。


大量に生成した画像の選り分けにも使えると思います。

## 使い方
### 0. データの用意
[sd-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)+[wd-14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)を使い、「Save with JSON」でスコア付きタグのjsonを作成する。
画像とjsonのファイル名は揃える。(保存場所は違っていてOK)

### 1. ツールを起動
ダウンロードなりcloneなりしてpythonで実行する。

`python main.py`

Pythonがなければ入れる。足りないライブラリがあったらpipする。

### 2. 画像をロード
「▼Files」を展開して画像とjsonのあるディレクトリ入力して「Load」する。
タグの.txtは任意。

### 3. 画像をフィルタリング
ウィンドウ左側で画像をソート/フィルタリングする。
フィルタは「{1girl} > 0.9」のように{タグ}とスコアの比較演算で書く。

### 4. スコア付きタグをフィルタリング
ウィンドウ右側でスコア付きタグをフィルタリングする。
正規表現使える。

### 5. 学習用タグを追加
「AddFilterdTags」でフィルタ条件に当てはまるタグを学習用タグとして追加する。
(中央のテキストボックスに表示される)。

対象の範囲は「Change For」で選ぶ。
学習用タグの書き換えや削除、置換も可能。


### 6. 学習用タグを保存
「▼Files」を展開して保存先を入力し「Dump Tags text」で、各画像の学習用タグのtxtを保存する。

[kohyaさんの学習スクリプト](https://github.com/kohya-ss/sd-scripts)で読み込める形式のjsonも出力可。

### 7. Let's Finetuning


## 機能
### 画像フィルタの詳細
* {タグ}のように括った部分がそのタグのスコアとして計算される。
* {}内のタグは正規表現でマッチ可能。複数マッチする場合は最大値をとる。
* pythonの文法で四則演算やand/or等が使用可能。
* 3つあるフィルタはそれぞれANDで適用される。

※フィルタ記述はevalで評価しているので注意

### 重複画像検出
「▼Display Option」→「DetectDuplicated」
タグのスコアを基準に重複画像を検出できる。

「Threshold」で類似度の閾値を設定可。
1.0が完全一致。0.97くらいが微妙な切り抜きや色調の差分等を無視して検出してくれるのでおすすめ。

「FilterDuplicated」で重複画像を表示から除外または重複画像のみ表示ができる。

### 画像のREMOVE
ビューア内のメタタグとして"REMOVED"をつけるだけで、実際の画像はいじらない。

「▼Display Option」項目でREMOVED画像の表示/非表示を切り替え可。
タグ保存の際には対象外になる。
