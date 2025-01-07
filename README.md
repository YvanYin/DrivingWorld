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
<br><br><image src="./images/pipeline.png"/>
</div>

We present **DrivingWorld** (World Model for Autonomous Driving), a model that enables autoregressive video and ego state generation with high efficiency. **DrivingWorld** formulates the future state prediction (ego state and visions) as a next-state autoregressive style. Our **DrivingWorld** is able to predict over 40s videos and achieves high-fidelity controllable generation.

## 🚀News

- ```[Dec 2024]``` Released [paper](https://arxiv.org/abs/2412.19505), inference codes, and quick start guide.

## 🔨 TODO LIST

- [ ] Hugging face demos
- [x] Complete evaluation code
- [x] Video preprocess code
- [ ] Training code


## ✨Hightlights

- 🔥 **Novel Approach**: GPT-style video and ego state generation.
- 🔥 **State-of-the-art Performance**:  and long-duration driving-scene video results.
- 🔥 **Controlable Generation**: High-fidelity controllable generation with ego poses.

## 🗄️Demos
- 🔥 Controllable generation with provided ego poses.
<a id="demo"></a>

<image src="./images/teaser.png"/>

![gif](https://raw.githubusercontent.com/huxiaotaostasy/huxiaotaostasy.github.io/main/DrivingWorld/videos/video_github.gif)

## 🙊 Model Zoo
| Model | Link |
|---|---|
|Video VQVAE| [link](https://huggingface.co/huxiaotaostasy/DrivingWorld/blob/main/vqvae.pt) |
|World Model| [link](https://huggingface.co/huxiaotaostasy/DrivingWorld/blob/main/world_model.pth) |


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

For data preparation, please refer to [video_data_preprocess.md](./video_data_preprocess//video_data_preprocess.md) for more details.


### Change Road Demo

Script for the default setting (conditioned on 15 frames, on demo videos, adopt topk sampling):
```bash
python3 tools/test_change_road_demo.py \
--config "configs/drivingworld_v1/gen_videovq_conf_demo.py" \
--exp_name "demo_dest_change_road" \
--load_path "./pretrained_models/world_model.pth" \
--save_video_path "./outputs/change_road"
```

### Long-term Demo

Script for the default setting (conditioned on 15 frames, on demo videos, adopt topk sampling):
```bash
python3 tools/test_long_term_demo.py \
--config "configs/drivingworld_v1/gen_videovq_conf_demo.py" \ 
--exp_name "demo_test_long_term" \
--load_path "./pretrained_models/world_model.pth" \
--save_video_path "./outputs/long_term"
```

### Personalized Generation

For all kinds of generation, you can change the conditional yaws and poses in the code yourself to get different outputs, and you can also modify the sampling parameters in the config files according to your needs.

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

