# Vision Transformer (ViT) ベースの画像キャプショニングモデルの例

## 概要

このモデルの例では，Hugging Face transformers ライブラリを使用して，画像キャプショニング（Image to Text）モデルを学習し，実際のキャプション生成を行います．

Hugging Face では，[Vision Encoder Decoder モデル](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder)  が提供されています．
このモデルは，任意の画像エンコーダと任意のテキストデコーダを組み合わせることができます．

この例では，ViT モデルをエンコーダとして使用し，GPT-2 モデルをデコーダとして使用します．

エンコーダには[google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k), 
デコーダには[rinna/japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium)を利用します．

## モデルの学習

### データの準備

学習データとして，
- [MS-COCO](https://cocodataset.org/) の train2014, val2014 の画像データ
- これらの画像に対して日本語でアノテーションされたキャプションデータである [STAIR Captions](http://captions.stair.center/)
を利用します．

これらはすでに diamond2 にダウンロードされています．

これからモデルの学習に使える形に変換するため， 以下のように [make_dataset.py](make_dataset.py) を実行してください．

```bash
$ python make_dataset.py
```

実行すると，`stair_captions_dataset` というディレクトリに学習に利用可能な形で保存されます（ディレクトリの中身のファイルは人では読めない形式になっています）．

### 学習の実行

学習は以下のように [train_model.py] を実行することで行えます．

```bash
$ python train_model.py
```

ただし，環境によってはメモリ不足で学習が途中で止まってしまうことがあります．
その場合は，プログラムの中の57行目付近にある `per_device_train_batch_size` の値を小さくしてください．

目安として, GPUのメモリが10GB程度の場合は2, 20GB程度で10を設定することができます．

計算機の性能やバッチサイズによると思いますが，学習には1日～数日程度かかります．

モデルの学習結果は, `models/vit-gpt2-japanese-image-captioning_stair-captions/` 以下に作成される,
`checkpoint-32600` などのディレクトリに保存されます．
ディレクトリ名には，学習時のステップ数が含まれていて，評価値の高いもの（評価データのロスが小さいもの）上位3つが保存されます．

## 学習されたモデルの利用

学習されたモデルを使って，実際の画像に対してキャプションを生成する例は [test_trained_model.ipynb](./test_trained_model.ipynb) を見てください．

2つ目のセルにある
```python
model = VisionEncoderDecoderModel.from_pretrained("./models/vit-gpt2-japanese-image-captioning_stair-captions/checkpoint-326000/")
```
のディレクトリ名は, 利用したいモデルのディレクトリ名に変更してください．
この際，ディレクトリ名にステップ数がついているのに注意してください．

`model.generaate()` の各種引数を変更することで，生成される文の数などをある程度コントロールすることが可能です．
詳しくは [Huggingface Transformers 入門 (6) - テキスト生成](https://note.com/npaka/n/n5d296d8ae26d) などを参照してください．











 