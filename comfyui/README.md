# ComfyUI CogVideoX-Fun
Easily use CogVideoX-Fun inside ComfyUI!

- [Installation](#1-installation)
- [Node types](#node-types)
- [Example workflows](#example-workflows)

## 1. Installation

### Option 1: Install via ComfyUI Manager
TBD

### Option 2: Install manually
The CogVideoX-Fun repository needs to be placed at `ComfyUI/custom_nodes/CogVideoX-Fun/`.

```
cd ComfyUI/custom_nodes/

# Git clone the cogvideox_fun itself
git clone https://github.com/aigc-apps/CogVideoX-Fun.git

# Git clone the video outout node
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

cd CogVideoX-Fun/
python install.py
```

### 2. Download models into `ComfyUI/models/CogVideoX-Fun/`

| Name | Storage Space | Url | Hugging Face | Description |
|--|--|--|--|--|
| CogVideoX-Fun-2b-InP.tar.gz | Before extraction:9.7 GB \/ After extraction: 13.0 GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/Diffusion_Transformer/CogVideoX-Fun-2b-InP.tar.gz) | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-2b-InP)| Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. |
| CogVideoX-Fun-5b-InP.tar.gz | Before extraction:16.0 GB \/ After extraction: 20.0 GB | [Download](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/Diffusion_Transformer/CogVideoX-Fun-5b-InP.tar.gz) | [🤗Link](https://huggingface.co/alibaba-pai/CogVideoX-Fun-5b-InP)| Our official graph-generated video model is capable of predicting videos at multiple resolutions (512, 768, 1024, 1280) and has been trained on 49 frames at a rate of 8 frames per second. |

## Node types
- **LoadCogVideoX_Fun_Model**
    - Loads the CogVideoX-Fun model
- **TextBox**
    - Write the prompt for CogVideoX-Fun model
- **CogVideoX_Fun_I2VSampler**
    - CogVideoX-Fun Sampler for Image to Video 
- **CogVideoX_Fun_T2VSampler**
    - CogVideoX-Fun Sampler for Text to Video
- **CogVideoX_Fun_V2VSampler**
    - CogVideoX-Fun Sampler for Video to Video

## Example workflows

### Video to video generation
Our ui is shown as follow, this is the [download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_v2v.json) of the json:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_v2v.jpg)

You can run the demo using following video:
[demo video](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/play_guitar.mp4)

### Image to video generation
Our ui is shown as follow, this is the [download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_i2v.json) of the json:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_i2v.jpg)

You can run the demo using following photo:
![demo image](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/firework.png)

### Text to video generation
Our ui is shown as follow, this is the [download link](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_t2v.json) of the json:
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/cogvideoxfunv1_workflow_t2v.jpg)