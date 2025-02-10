# Goku: Flow Based Video Generative Foundation Models

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2502.04896-b31b1b.svg)](https://arxiv.org/abs/2502.04896)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://saiyan-world.github.io/goku/)&nbsp;
  
</div>

<div align="center">
<video src="https://github.com/user-attachments/assets/1dc60a41-b8ff-4bfd-bfba-3a185ae63345" width="100%" controls autoplay loop></video>
</div>


> [**Goku: Flow Based Video Generative Foundation Models**](https://arxiv.org/abs/2502.04896)<br>
> [Shoufa Chen](https://www.shoufachen.com), [Chongjian Ge](https://chongjiange.github.io/), [Yuqi Zhang](https://scholar.google.com/citations?user=7FlkVy8AAAAJ), [Yida Zhang](https://openreview.net/profile?id=~Yida_Zhang2), [Fengda Zhu](https://www.zhufengda.net/), [Hao Yang](https://github.com/haoy945), [Hongxiang Hao](https://scholar.google.com/citations?user=173GpBQAAAAJ&hl=zh-CN), [Hui Wu](https://github.com/whlook), [Zhichao Lai](https://github.com/lazychao), [Yifei Hu](https://openreview.net/profile?id=~Yifei_Hu3), [Ting-Che Lin](https://github.com/tcl326), [Shilong Zhang](https://jshilong.github.io/), [Fu Li](https://scholar.google.com/citations?user=2A7_3hoAAAAJ&hl=en), [Chuan Li](https://www.linkedin.com/in/chuanli1101/), [Xing Wang](https://www.linkedin.com/in/xing-wang-49369620/), [Yanghua Peng](https://scholar.google.com/citations?user=Gf9amnoAAAAJ&hl=en), [Peize Sun](https://peizesun.github.io/), [Ping Luo](http://luoping.me/), [Yi Jiang](https://scholar.google.com/citations?user=6dikuoYAAAAJ&hl=en), [Zehuan Yuan](https://shallowyuan.github.io/), [Bingyue Peng](https://www.linkedin.com/in/bingyp), [Xiaobing Liu](https://scholar.google.com/citations?user=1ypDmDwAAAAJ&hl=en)
> <br>HKU, ByteDance<br>


## Overview 
Goku is a new family of joint image-and-video generation models based on rectified flow Transformers. It is designed to achieve industry-grade performance, integrating advanced techniques for high-quality visual generation, including meticulous data curation, model design, and flow formulation.

Key contributions include:
- 📊 High-quality fine-grained image and video data curation.
- 🔄 The pioneering use of rectified flow for enhanced interaction among video and image tokens.
- 🌟 Superior qualitative and quantitative performance in both image and video generation tasks.

Goku supports multiple generation tasks:
- 🎬 **Text-to-Video Generation**
- 🖼️ **Image-to-Video Generation**
- 🎨 **Text-to-Image Generation**

## Performance Benchmarks 🏅
Goku achieves top scores on major benchmarks:
- **0.76** on GenEval (text-to-image generation) 
- **83.65** on DPG-Bench (text-to-image generation) 
- **84.85** on VBench (text-to-video generation) 





### VBench Performance 🏆
Goku-T2V achieves an impressive score of **84.85** in VBench, securing the No.2 position as of 2024-10-07, surpassing several leading commercial text-to-video models.

| Method         | Total Score | Quality Score | Sampling Score | Style Consistency | Background Consistency | Temporal Flickering | Motion Smoothness | Dynamic Degree | Subject Quality | Imaging Quality | Object Class | Human Action | Object Relationship | Color | Scene | Prompt Style | Overall Consistency |
|---------------|-------------|--------------|----------------|---------------------|---------------------|-----------------|----------------|-----------------|---------------|---------------|-------------|---------------|------------------|-------|------|-------------|----------------|
| **AnimateDiff-V2** | 80.27 | 82.90 | 69.75 | 95.30 | 97.68 | 98.75 | 97.76 | 40.83 | 67.16 | 70.10 | 90.90 | 36.88 | 92.60 | 87.47 | 34.60 | 50.19 | 22.42 | 26.03 | 27.04 |
| **VideoCrafter-2.0** | 80.44 | 82.20 | 73.42 | 96.85 | **98.22** | 98.41 | 97.73 | 42.50 | 63.13 | 67.22 | 92.55 | 40.66 | 95.00 | **92.92** | 35.86 | 55.29 | **25.13** | 25.84 | **28.23** |
| **OpenSora V1.2** | 79.23 | 80.71 | 73.30 | 94.45 | 97.90 | 99.47 | 98.20 | 47.22 | 56.18 | 60.94 | 83.37 | 58.41 | 85.80 | 87.49 | 67.51 | 42.47 | 23.89 | 24.55 | 27.07 |
| **Show-1** | 78.93 | 80.42 | 72.98 | 95.53 | 98.02 | 99.12 | 98.24 | 44.44 | 57.35 | 58.66 | 93.07 | 45.47 | 95.60 | 86.35 | 53.50 | 47.03 | 23.06 | 25.28 | 27.46 |
| **Gen-3** | 82.32 | 84.11 | 75.17 | 97.10 | 96.62 | 98.61 | 99.23 | 60.14 | 63.34 | 66.82 | 87.81 | 53.64 | 96.40 | 80.90 | 65.09 | 54.57 | 24.31 | 24.71 | 26.69 |
| **Pika-1.0** | 80.69 | 82.92 | 71.77 | 96.94 | 97.36 | **99.74** | **99.50** | 47.50 | 62.04 | 61.87 | 88.72 | 43.08 | 86.20 | 90.57 | 61.03 | 49.83 | 22.26 | 24.22 | 25.94 |
| **CogVideoX-5B** | 81.61 | 82.75 | 77.04 | 96.23 | 96.52 | 98.66 | 96.92 | 70.97 | 61.98 | 62.90 | 85.23 | 62.11 | 99.40 | 82.81 | 66.35 | 53.20 | 24.91 | 25.38 | 27.59 |
| **Kling** | 81.85 | 83.39 | 75.68 | **98.33** | 97.60 | 99.30 | 99.40 | 46.94 | 61.21 | 65.62 | 87.24 | 68.05 | 93.40 | 89.90 | 73.03 | 50.86 | 19.62 | 24.17 | 26.42 |
| **Mira** | 71.87 | 78.78 | 44.21 | 96.23 | 96.92 | 98.29 | 97.54 | 60.33 | 42.51 | 60.16 | 52.06 | 12.52 | 63.80 | 42.24 | 27.83 | 16.34 | 21.89 | 18.77 | 18.72 |
| **CausVid** | 84.27 | **85.65** | 78.75 | 97.53 | 97.19 | 96.24 | 98.05 | **92.69** | 64.15 | 68.88 | 92.99 | 72.15 | **99.80** | 80.17 | 64.65 | 56.58 | 24.27 | 25.33 | 27.51 |
| **Luma** | 83.61 | 83.47 | **84.17** | 97.33 | 97.43 | 98.64 | 99.35 | 44.26 | 65.51 | 66.55 | **94.95** | **82.63** | 96.40 | 92.33 | 83.67 | **58.98** | 24.66 | **26.29** | 28.13 |
| **HunyuanVideo** | 83.24 | 85.09 | 75.82 | 97.37 | 97.76 | 99.44 | 98.99 | 70.83 | 60.36 | 67.56 | 86.10 | 68.55 | 94.40 | 91.60 | 68.68 | 53.88 | 19.80 | 23.89 | 26.44 |
| **Goku-T2V** (ours) | 84.85 | 85.60 | 81.87 | 95.55 | 96.67 | 97.71 | 98.50 | 76.11 | 67.22 | 71.29 | 94.40 | 79.48 | 97.60 | 83.81 | 85.72 | 57.08 | 23.08 | 25.64 | 27.35 |



## BibTeX
```bibtex
@article{chen2025goku,
  title={Goku: Flow Based Video Generative Foundation Models},
  author={Chen, Shoufa and Ge, Chongjian and Zhang, Yuqi and Zhang, Yida and Zhu, Fengda and Yang, Hao and Hao, Hongxiang and Wu, Hui and Lai, Zhichao and Hu, Yifei and Lin, Ting-Che and Zhang, Shilong and Li, Fu and Li, Chuan and Wang, Xing and Peng, Yanghua and Sun, Peize and Luo, Ping and Jiang, Yi and Yuan, Zehuan and Peng, Bingyue and Liu, Xiaobing},
  journal={arXiv preprint arXiv:2502.04896},
  year={2025}
}
```
