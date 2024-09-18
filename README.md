# CogVideoX-Fun

😊 Welcome!

English | [简体中文](./README_zh-CN.md)

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Video Result](#video-result)
- [How to use](#how-to-use)
- [Model zoo](#model-zoo)
- [TODO List](#todo-list)
- [Reference](#reference)
- [License](#license)

# Introduction
CogVideoX-Fun is a modified pipeline based on the CogVideoX structure, designed to provide more flexibility in generation. It can be used to create AI images and videos, as well as to train baseline models and Lora models for Diffusion Transformer. We support predictions directly from the already trained CogVideoX-Fun model, allowing the generation of videos at different resolutions, approximately 6 seconds long with 8 fps (1 to 49 frames). Users can also train their own baseline models and Lora models to achieve certain style transformations.

We will support quick pull-ups from different platforms, refer to [Quick Start](#quick-start).

What's New:
- Create code! Now supporting Windows and Linux. Supports 2b and 5b models. Supports video generation at any resolution from 256x256x49 to 1024x1024x49. [ 2024.09.18 ]

Function：
- [Data Preprocessing](#data-preprocess)
- [Train DiT](#dit-train)
- [Video Generation](#video-gen)

Our UI interface is as follows:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/ui.jpg)

# Quick Start
### 1. Cloud usage: AliyunDSW/Docker
#### a. From AliyunDSW
DSW has free GPU time, which can be applied once by a user and is valid for 3 months after applying.

Aliyun provide free GPU time in [Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1), get it and use in Aliyun PAI-DSW to start CogVideoX-Fun within 5min!

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/cogvideox_fun)

#### b. From ComfyUI
Our ComfyUI is as follows, please refer to [ComfyUI README](comfyui/README.md) for details.
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_i2v.jpg)

#### c. From docker
If you are using docker, please make sure that the graphics card driver and CUDA environment have been installed correctly in your machine.

Then execute the following commands in this way:

```
# pull image
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# enter image
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:cogvideox_fun

# clone code
git clone https://github.com/aigc-apps/CogVideoX-Fun.git

# enter CogVideoX-Fun's dir
cd CogVideoX-Fun

# download weights
mkdir models/Diffusion_Transformer
mkdir models/Personalized_Model

wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/Diffusion_Transformer/CogVideoX-Fun-2b-InP.tar.gz -O models/Diffusion_Transformer/CogVideoX-Fun-2b-InP.tar.gz
wget https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/Diffusion_Transformer/CogVideoX-Fun-5b-InP.tar.gz -O models/Diffusion_Transformer/CogVideoX-Fun-5b-InP.tar.gz

cd models/Diffusion_Transformer/
tar -xvf CogVideoX-Fun-2b-InP.tar.gz
tar -xvf CogVideoX-Fun-5b-InP.tar.gz
cd ../../
```

### 2. Local install: Environment Check/Downloading/Installation
#### a. Environment Check
We have verified CogVideoX-Fun execution on the following environment:

The detailed of Windows:
- OS: Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G & Nvidia-3090 24G

The detailed of Linux:
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

We need about 60GB available on disk (for saving weights), please check!

#### b. Weights
We'd better place the [weights](#model-zoo) along the specified path:

```
📦 models/
├── 📂 Diffusion_Transformer/
│   ├── 📂 CogVideoX-Fun-2b-InP/
│   └── 📂 CogVideoX-Fun-5b-InP/
├── 📂 Personalized_Model/
│   └── your trained trainformer model / your trained lora model (for UI load)
```

# Video Result
The results displayed are all based on image. 

### CogVideoX-Fun-5B

Resolution-1024

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/ec749326-b529-453f-a4b4-f587875dff64" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/84df4178-f493-4aa8-a888-d2020338da82" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/c66c139d-94d3-4930-985b-60e3e0600d8f" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/647c0e0c-28d6-473e-b4eb-a30197dddefc" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

Resolution-768

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/647d45b0-4253-4438-baf3-f692789bde78" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/e5a5a948-5c34-445d-9446-324a666a6a33" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/0e605797-4a86-4e0c-8589-40ed686d97a4" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/5356bf79-0a3b-4caf-ac31-2d796e20e429" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

Resolution-512

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/5a9f3457-fe82-4082-8494-d8f4f8db75e9" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ca6874b8-41d1-4f02-bee3-4fc886f309ad" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/9216b348-2c80-4eab-9c1c-dd3a54b7ea1e" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/e99ec495-655f-44d8-afa7-3ad0a14f9975" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### CogVideoX-Fun-2B

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/d329b4d4-f08f-4e77-887e-049cfc93a908" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/dd7fa2d5-9871-436c-ae5a-44f1494c9c9f" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/c24a2fa2-2fe3-4277-aa9f-e812a2cf0a4e" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/573edac3-8bd0-4e95-82df-bcfdcba9a73f" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

# How to use

<h3 id="video-gen">1. Inference </h3>

#### a. Using Python Code
- Step 1: Download the corresponding [weights](#model-zoo) and place them in the models folder.
- Step 2: Modify prompt, neg_prompt, guidance_scale, and seed in the predict_t2v.py file.
- Step 3: Run the predict_t2v.py file, wait for the generated results, and save the results in the samples/cogvideox-fun-videos-t2v folder.
- Step 4: If you want to combine other backbones you have trained with Lora, modify the predict_t2v.py and Lora_path in predict_t2v.py depending on the situation.

#### b. Using webui
- Step 1: Download the corresponding [weights](#model-zoo) and place them in the models folder.
- Step 2: Run the app.py file to enter the graph page.
- Step 3: Select the generated model based on the page, fill in prompt, neg_prompt, guidance_scale, and seed, click on generate, wait for the generated result, and save the result in the samples folder.

#### c. From ComfyUI
Please refer to [ComfyUI README](comfyui/README.md) for details.

### 2. Model Training
A complete CogVideoX-Fun training pipeline should include data preprocessing, and Video DiT training. 

<h4 id="data-preprocess">a. data preprocessing</h4>

We have provided a simple demo of training the Lora model through image data, which can be found in the [wiki](https://github.com/aigc-apps/CogVideoX-Fun/wiki/Training-Lora) for details.

A complete data preprocessing link for long video segmentation, cleaning, and description can refer to [README](cogvideox/video_caption/README.md) in the video captions section. 

If you want to train a text to image and video generation model. You need to arrange the dataset in this format.

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

The json_of_internal_datasets.json is a standard JSON file. The file_path in the json can to be set as relative path, as shown in below:
```json
[
    {
      "file_path": "train/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

You can also set the path as absolute path as follow:
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

<h4 id="dit-train">b. Video DiT training </h4>
 
If the data format is relative path during data preprocessing, please set ```scripts/train.sh``` as follow.
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

If the data format is absolute path during data preprocessing, please set ```scripts/train.sh``` as follow.
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

Then, we run scripts/train.sh.
```sh
sh scripts/train.sh
```

For details on setting some parameters, please refer to [Readme Train](scripts/README_TRAIN.md) and [Readme Lora](scripts/README_TRAIN_LORA.md). 


# Model zoo

| Name | Storage Space | Url | Hugging Face | Description |
|--|--|--|--|--|
| CogVideoX-Fun-2b-InP.tar.gz | Before extraction:9.7 GB \/ After extraction: 13.0 GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/Diffusion_Transformer/CogVideoX-Fun-2b-InP.tar.gz) | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-2b-InP)| Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 24 frames per second. |
| CogVideoX-Fun-5b-InP.tar.gz | Before extraction:16.0 GB \/ After extraction: 20.0 GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/Diffusion_Transformer/CogVideoX-Fun-5b-InP.tar.gz) | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-5b-InP)| Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 24 frames per second. |

# TODO List
- Support Chinese.

# Reference
- CogVideo: https://github.com/THUDM/CogVideo/
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate

# License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

The CogVideoX-2B model (including its corresponding Transformers module and VAE module) is released under the [Apache 2.0 License](LICENSE).

The CogVideoX-5B model (Transformers module) is released under the [CogVideoX LICENSE](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE).
