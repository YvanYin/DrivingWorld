<div align="center">

<h1>DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT</h1>

<p align="center">
<a href="https://arxiv.org/abs/2412.19505"><img src="https://img.shields.io/badge/ArXiv-2412.19505-%23840707.svg" alt="ArXiv"></a>
<a href="https://youtu.be/5QJRAxnjX0k"><img src="https://img.shields.io/badge/Youtube Demo-Video-%26840707.svg" alt="VideoDemo"></a>
<a href="https://huxiaotaostasy.github.io/DrivingWorld/index.html"><img src="https://img.shields.io/badge/Webpage-DrivingWorld-%237CB4F7.svg" alt="Webpage"></a>
</p>

[Xiaotao Hu](https://huxiaotaostasy.github.io/)<sup>1,2*</sup>, [Wei Yin](https://yvanyin.net/)<sup>2*§</sup>, [Mingkai Jia](https://scholar.google.com/citations?user=fcpTdvcAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Junyuan Deng](https://scholar.google.com/citations?user=KTCPC5IAAAAJ&hl=en)<sup>1,2</sup>, [Xiaoyang Guo](https://xy-guo.github.io/)<sup>2</sup><br>
[Qian Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=pCY-bikAAAAJ)<sup>2</sup>, [Xiaoxiao Long](https://www.xxlong.site/)<sup>1†</sup>, [Ping Tan](https://scholar.google.com/citations?user=XhyKVFMAAAAJ&hl=en)<sup>1</sup><br>

[HKUST](https://hkust.edu.hk/)<sup>1</sup>, [Horizon Robotics](https://en.horizon.auto/)<sup>2</sup><br>
<sup>*</sup> Equal Contribution, <sup>†</sup> Corresponding Author, <sup>§</sup> Project Leader
<br><br><image src="./images/teaser.png"/>
</div>

We present **DrivingWorld** (World Model for Autonomous Driving), a model that enables autoregressive video and ego state generation with high efficiency. **DrivingWorld** formulates the future state prediction (ego state and visions) as an next-state autoregressive prediction. **DrivingWorld** enables generate over 40s videos.

## 🚀News

- ```[Dec 2024]``` Released [paper](https://arxiv.org/abs/2412.19505), inference codes, and Quick Start guide.

## 🔨 TODO LIST

- [ ] Hugging face demos
- [ ] More demos
- [ ] Complete evaluation code
- [ ] Video Preprocess Code
- [ ] Training code


## ✨Hightlights

- 🔥 **Novel Approach**: GPT-style video and ego state generation.
- 🔥 **State-of-the-art Performance**:  and long-duration driving-scene video results.
- 🔥 **Controlable Generation**: High-fidelity controllable generation with ego poses.

## 🗄️Demos

<a id="demo"></a>
![gif](https://raw.githubusercontent.com/huxiaotaostasy/huxiaotaostasy.github.io/main/DrivingWorld/videos/video_github.gif)



## 🔑 Quick Start
<a id="quick start"></a>

### Installation

```bash
git clone https://github.com/YvanYin/DrivingWorld.git
cd DrivingWorld
pip3 install -r requirements.txt
```
* Download the pretrained models from [Hugging Face](https://huggingface.co/huxiaotaostasy/DrivingWorld/tree/main), and move the pretrained parameters to `DrivingWorld/pretrained_models/*`

### Data Preparation
For the public dataset, we use [NuPlan](https://nuplan.org/) and [OpenDV-YouTube](https://github.com/OpenDriveLab/DriveAGI?tab=readme-ov-file#opendv) for testing.

We download `nuPlan Val Split` in [NuPlan](https://nuplan.org/). And we follow [OpenDV-YouTube](https://github.com/OpenDriveLab/DriveAGI/blob/main/opendv/README.md) to get the validation set.

We share the `json` files in [Hugging Face](https://huggingface.co/huxiaotaostasy/DrivingWorld/tree/main).


### Evaluation 
Script for the default setting (conditioned on 15 frames, on Nuplan Validation set, adopt topk sampling):
```bash
cd tools
sh demo_test_long_term_nuplan.sh
sh demo_test_long_term_youtube.sh
sh demo_test_change_road.sh
```
You can change the setting with config file in \<configs/\>



## 📌 Citation

If the paper and code from `DrivingWorld` help your research, we kindly ask you to give a citation to our paper ❤️. Additionally, if you appreciate our work and find this repository useful, giving it a star ⭐️ would be a wonderful way to support our work. Thank you very much.

```bibtex
@article{hu2024drivingworld,
  title={DrivingWorld: ConstructingWorld Model for Autonomous Driving via Video GPT},
  author={Hu, Xiaotao and Yin, Wei and Jia, Mingkai and Deng, Junyuan and Guo, Xiaoyang and Zhang, Qian and Long, Xiaoxiao and Tan, Ping},
  journal={arXiv preprint arXiv:2412.19505},
  year={2024}
}
```

## Reference
We thank for [VQGAN](https://github.com/CompVis/taming-transformers), [LlamaGen](https://github.com/FoundationVision/LlamaGen) and [LLlama 3.1](https://github.com/meta-llama/llama3) for their codebase.



## License

This repository is under the MIT License. For more license questions, please contact Wei Yin (yvanwy@outlook.com) and Xiaotao Hu (xiaotao.hu@connect.ust.hk).

