# CogVideoX-Fun

😊 ようこそ！

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/CogVideoX-Fun-5b)

[English](./README.md) | [简体中文](./README_zh-CN.md) | 日本語

# 目次
- [目次](#目次)
- [紹介](#紹介)
- [クイックスタート](#クイックスタート)
- [ビデオ結果](#ビデオ結果)
- [使用方法](#使用方法)
- [モデルの場所](#モデルの場所)
- [TODOリスト](#todoリスト)
- [参考文献](#参考文献)
- [ライセンス](#ライセンス)

# 紹介
CogVideoX-Funは、CogVideoX構造に基づいて修正されたパイプラインであり、生成の柔軟性を提供します。AI画像とビデオの生成、Diffusion TransformerのベースラインモデルとLoraモデルのトレーニングに使用できます。すでにトレーニングされたCogVideoX-Funモデルから直接予測を行い、異なる解像度で約6秒間、8fps（1〜49フレーム）のビデオを生成できます。ユーザーは独自のベースラインモデルとLoraモデルをトレーニングして、特定のスタイル変換を実現することもできます。

異なるプラットフォームからのクイックスタートをサポートします。詳細は[クイックスタート](#クイックスタート)を参照してください。

新機能：
- I2Vモデルと関連するトレーニング予測コードをV1.5バージョンに更新。[2024.12.16]
- 報酬逆伝播を使用してLoraをトレーニングし、生成されたビデオを最適化し、人間の好みにより一致させます。詳細は[こちら](scripts/README_TRAIN_REWARD.md)をご覧ください。新しいバージョンのコントロールモデルは、Canny、Depth、Pose、MLSDなどのさまざまな条件をサポートします。[2024.11.21]
- CogVideoX-Fun Controlはdiffusersでサポートされています。サポートを提供してくれた[a-r-r-o-w](https://github.com/a-r-r-o-w)に感謝します。この[PR](https://github.com/huggingface/diffusers/pull/9671)でサポートが提供されました。詳細は[ドキュメント](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox)をご覧ください。[2024.10.16]
- i2vモデルを再トレーニングし、ノイズを追加してビデオの動きの幅を広げました。コントロールモデルのトレーニングコードとコントロールモデルをアップロードしました。[2024.09.29]
- コードを作成しました！現在、WindowsとLinuxをサポートしています。2bおよび5bモデルをサポートし、256x256x49から1024x1024x49までの任意の解像度でビデオを生成できます。[2024.09.18]

機能：
- [データ前処理](#data-preprocess)
- [DiTのトレーニング](#dit-train)
- [ビデオ生成](#video-gen)

私たちのUIインターフェースは次のとおりです：
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/ui.jpg)

# クイックスタート
### 1. クラウド使用: AliyunDSW/Docker
#### a. AliyunDSWから
DSWには無料のGPU時間があり、ユーザーは一度申請でき、申請後3か月間有効です。

Aliyunは[Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1)で無料のGPU時間を提供しています。取得してAliyun PAI-DSWで使用し、5分以内にCogVideoX-Funを開始できます！

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/cogvideox_fun)

#### b. ComfyUIから
私たちのComfyUIは次のとおりです。詳細は[ComfyUI README](comfyui/README.md)を参照してください。
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_i2v.jpg)

#### c. Dockerから
Dockerを使用する場合、マシンにグラフィックスカードドライバとCUDA環境が正しくインストールされていることを確認してください。

次のコマンドをこの方法で実行します：

```
# イメージをプル
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# イメージに入る
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# コードをクローン
git clone https://github.com/aigc-apps/CogVideoX-Fun.git

# CogVideoX-Funのディレクトリに入る
cd CogVideoX-Fun

# 重みをダウンロード
mkdir models/Diffusion_Transformer
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP.tar.gz -O models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP.tar.gz

cd models/Diffusion_Transformer/
tar -xvf CogVideoX-Fun-V1.1-2b-InP.tar.gz
cd ../../
```

### 2. ローカルインストール: 環境チェック/ダウンロード/インストール
#### a. 環境チェック
以下の環境でCogVideoX-Funの実行を確認しました：

Windowsの詳細：
- OS: Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G & Nvidia-3090 24G

Linuxの詳細：
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

重みを保存するために約60GBのディスクスペースが必要です。確認してください！

#### b. 重み
[重み](#model-zoo)を指定されたパスに配置することをお勧めします：

```
📦 models/
├── 📂 Diffusion_Transformer/
│   ├── 📂 CogVideoX-Fun-V1.1-2b-InP/
│   └── 📂 CogVideoX-Fun-V1.1-5b-InP/
├── 📂 Personalized_Model/
│   └── あなたのトレーニング済みのトランスフォーマーモデル / あなたのトレーニング済みのLoraモデル（UIロード用）
```

# ビデオ結果
表示されている結果はすべて画像に基づいています。

### CogVideoX-Fun-V1.1-5B

解像度-1024

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/34e7ec8f-293e-4655-bb14-5e1ee476f788" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7809c64f-eb8c-48a9-8bdc-ca9261fd5434" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/8e76aaa4-c602-44ac-bcb4-8b24b72c386c" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/19dba894-7c35-4f25-b15c-384167ab3b03" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>


解像度-768

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0bc339b9-455b-44fd-8917-80272d702737" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/70a043b9-6721-4bd9-be47-78b7ec5c27e9" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d5dd6c09-14f3-40f8-8b6d-91e26519b8ac" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9327e8bc-4f17-46b0-b50d-38c250a9483a" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

解像度-512

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ef407030-8062-454d-aba3-131c21e6b58c" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/7610f49e-38b6-4214-aa48-723ae4d1b07e" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1fff0567-1e15-415c-941e-53ee8ae2c841" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/bcec48da-b91b-43a0-9d50-cf026e00fa4f" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-5B with Reward Backpropagation

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="10%">プロンプト</sup></th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-5B</th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-5B <br> HPSv2.1 Reward LoRA</th>
            <th style="text-align: center;" width="30%">CogVideoX-Fun-V1.1-5B <br> MPS Reward LoRA</th>
        </tr>
    </thead>
    <tr>
        <td>
            翼のある豚がダイヤモンドの山の上を飛んでいる
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/6682f507-4ca2-45e9-9d76-86e2d709efb3" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/ec9219a2-96b3-44dd-b918-8176b2beb3b0" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/a75c6a6a-0b69-4448-afc0-fda3c7955ba0" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            犬が野原を走り、猫が木に登る
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/0392d632-2ec3-46b4-8867-0da1db577b6d" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/7d8c729d-6afb-408e-b812-67c40c3aaa96" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/dcd1343c-7435-4558-b602-9c0fa08cbd59" width="100%" controls autoplay loop></video>
        </td>
    </tr>
</table>

### CogVideoX-Fun-V1.1-5B-Control

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/53002ce2-dd18-4d4f-8135-b6f68364cabd" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fce43c0b-81fa-4ab2-9ca7-78d786f520e6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/b208b92c-5add-4ece-a200-3dbbe47b93c3" width="100%" controls autoplay loop></video>
     </td>
  <tr>
      <td>
          美しい澄んだ目と金髪の若い女性が白い服を着て体をひねり、カメラは彼女の顔に焦点を合わせています。高品質、傑作、最高品質、高解像度、超微細、夢のような。
      </td>
      <td>
          美しい澄んだ目と金髪の若い女性が白い服を着て体をひねり、カメラは彼女の顔に焦点を合わせています。高品質、傑作、最高品質、高解像度、超微細、夢のような。
      </td>
       <td>
          若いクマ。
     </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ea908454-684b-4d60-b562-3db229a250a9" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ffb7c6fc-8b69-453b-8aad-70dfae3899b9" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/d3f757a3-3551-4dcb-9372-7a61469813f5" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-5B-Pose

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          解像度-512
      </td>
      <td>
          解像度-768
      </td>
       <td>
          解像度-1024
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/a746df51-9eb7-4446-bee5-2ee30285c143" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/db295245-e6aa-43be-8c81-32cb411f1473" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/ec9875b2-fde0-48e1-ab7e-490cee51ef40" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-2B

解像度-768

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/03235dea-980e-4fc5-9c41-e40a5bc1b6d0" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/f7302648-5017-47db-bdeb-4d893e620b37" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/cbadf411-28fa-4b87-813d-da63ff481904" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/87cc9d0b-b6fe-4d2d-b447-174513d169ab" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-V1.1-2B-Pose

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          解像度-512
      </td>
      <td>
          解像度-768
      </td>
       <td>
          解像度-1024
      </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/487bcd7b-1b7f-4bb4-95b5-96a6b6548b3e" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/2710fd18-8489-46e4-8086-c237309ae7f6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/b79513db-7747-4512-b86c-94f9ca447fe2" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

# 使用方法

<h3 id="video-gen">1. 推論 </h3>

#### a. Pythonコードを使用
- ステップ1: 対応する[重み](#model-zoo)をダウンロードし、modelsフォルダに配置します。
- ステップ2: predict_t2v.pyファイルでprompt、neg_prompt、guidance_scale、およびseedを変更します。
- ステップ3: predict_t2v.pyファイルを実行し、生成された結果を待ちます。結果はsamples/cogvideox-fun-videos-t2vフォルダに保存されます。
- ステップ4: 他のトレーニング済みのバックボーンとLoraを組み合わせたい場合は、状況に応じてpredict_t2v.pyおよびLora_pathを変更します。

#### b. WebUIを使用
- ステップ1: 対応する[重み](#model-zoo)をダウンロードし、modelsフォルダに配置します。
- ステップ2: app.pyファイルを実行してグラフページに入ります。
- ステップ3: ページに基づいて生成モデルを選択し、prompt、neg_prompt、guidance_scale、およびseedを入力し、生成をクリックして生成結果を待ちます。結果はsamplesフォルダに保存されます。

#### c. ComfyUIから
詳細は[ComfyUI README](comfyui/README.md)を参照してください。

### 2. モデルのトレーニング
完全なCogVideoX-Funトレーニングパイプラインには、データ前処理とVideo DiTトレーニングが含まれている必要があります。

<h4 id="data-preprocess">a. データ前処理</h4>

画像データを使用してLoraモデルをトレーニングする簡単なデモを提供しました。詳細は[wiki](https://github.com/aigc-apps/CogVideoX-Fun/wiki/Training-Lora)をご覧ください。

長いビデオのセグメンテーション、クリーニング、説明のための完全なデータ前処理リンクは、ビデオキャプションセクションの[README](cogvideox/video_caption/README.md)を参照してください。

テキストから画像およびビデオ生成モデルをトレーニングしたい場合。この形式でデータセットを配置する必要があります。

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.jsonは標準のJSONファイルです。json内のfile_pathは相対パスとして設定できます。以下のように：
```json
[
    {
      "file_path": "train/00000001.mp4",
      "text": "スーツとサングラスを着た若い男性のグループが街の通りを歩いている。",
      "type": "video"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "スーツとサングラスを着た若い男性のグループが街の通りを歩いている。",
      "type": "image"
    },
    .....
]
```

次のように絶対パスとして設定することもできます：
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "スーツとサングラスを着た若い男性のグループが街の通りを歩いている。",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "スーツとサングラスを着た若い男性のグループが街の通りを歩いている。",
      "type": "image"
    },
    .....
]
```

<h4 id="dit-train">b. Video DiTトレーニング </h4>

データ前処理時にデータ形式が相対パスの場合、```scripts/train.sh```を次のように設定します。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

データ形式が絶対パスの場合、```scripts/train.sh```を次のように設定します。
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

次に、scripts/train.shを実行します。
```sh
sh scripts/train.sh
```

一部のパラメータの設定の詳細については、[Readme Train](scripts/README_TRAIN.md)、[Readme Lora](scripts/README_TRAIN_LORA.md)、および[Readme Control](scripts/README_TRAIN_CONTROL.md)を参照してください。

# モデルの場所

V1.5:

| 名称 | ストレージスペース | Hugging Face | Model Scope | 説明 |
|--|--|--|--|--|
| CogVideoX-Fun-V1.5-5b-InP |  20.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-5b-InP) | 公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024）でビデオを予測できます。85フレーム、8フレーム/秒でトレーニングされています。 |
| CogVideoX-Fun-V1.5-Reward-LoRAs | - | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs) | [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-Reward-LoRAs) | 公式の報酬逆伝播技術モデルで、CogVideoX-Fun-V1.5が生成するビデオを最適化し、人間の嗜好によりよく合うようにする。 |

V1.1:

| 名称 | ストレージスペース | Hugging Face | Model Scope | 説明 |
|--|--|--|--|--|
| CogVideoX-Fun-V1.1-2b-InP |  13.0 GB | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-InP) | [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-InP) | 公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。参照画像にノイズが追加され、V1.0と比較して動きの幅が広がっています。 |
| CogVideoX-Fun-V1.1-5b-InP |  20.0 GB  | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP) | [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-InP) | 公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。参照画像にノイズが追加され、V1.0と比較して動きの幅が広がっています。 |
| CogVideoX-Fun-V1.1-2b-Pose |  13.0 GB | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose) | [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Pose) | 公式のポーズコントロールビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。|
| CogVideoX-Fun-V1.1-2b-Control | 13.0 GB  | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Control) | [😄Link](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-2b-Control) | 公式のコントロールビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。Canny、Depth、Pose、MLSDなどのさまざまなコントロール条件をサポートします。|
| CogVideoX-Fun-V1.1-5b-Pose |  20.0 GB  | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose) | [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Pose) | 公式のポーズコントロールビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。|
| CogVideoX-Fun-V1.1-5b-Control |  20.0 GB  | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Control) | [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.1-5b-Control) | 公式のコントロールビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。Canny、Depth、Pose、MLSDなどのさまざまなコントロール条件をサポートします。|
| CogVideoX-Fun-V1.1-Reward-LoRAs | - | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-Reward-LoRAs) | [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-V1.5-Reward-LoRAs) | 公式の報酬逆伝播技術モデルで、CogVideoX-Fun-V1.1が生成するビデオを最適化し、人間の嗜好によりよく合うようにする。 |

V1.0:

| 名称 | ストレージスペース | Hugging Face | Model Scope | 説明 |
|--|--|--|--|--|
| CogVideoX-Fun-2b-InP |  13.0 GB | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-2b-InP) | [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-2b-InP) | 公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。 |
| CogVideoX-Fun-5b-InP |  20.0 GB  | [🤗リンク](https://huggingface.co/alibaba-pai/CogVideoX-Fun-5b-InP)| [😄リンク](https://modelscope.cn/models/PAI/CogVideoX-Fun-5b-InP)| 公式のグラフ生成ビデオモデルは、複数の解像度（512、768、1024、1280）でビデオを予測できます。49フレーム、8フレーム/秒でトレーニングされています。|

# TODOリスト
- 日本語をサポート。

# 参考文献
- CogVideo: https://github.com/THUDM/CogVideo/
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate

# ライセンス
このプロジェクトは[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)の下でライセンスされています。

CogVideoX-2Bモデル（対応するTransformersモジュール、VAEモジュールを含む）は、[Apache 2.0ライセンス](LICENSE)の下でリリースされています。

CogVideoX-5Bモデル（Transformersモジュール）は、[CogVideoXライセンス](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE)の下でリリースされています。
