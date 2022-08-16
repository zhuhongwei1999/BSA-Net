## I can find you! Boundary-guided Separated Attention Network for Camouflaged Object Detection (AAAI-22)

This repo is an official implementation of BSA-Net,  which has been published in 36th IEEE Conference on Artificial Intelligence (AAAI-22).

>Authors: Hongwei Zhu, Peng Li, Haoran Xie, Xuefeng Yan, Dong Liang, Dapeng Chen, Mingqiang Wei and Jing Qin

[toc]

## Overview

### Intro

The main pipeline of our BSA-Net is shown as the following,

![](C:\Users\zhw\Desktop\BSANet\figure\pipeline.png)

BSA-Net simulates the procedure of how humans to detect camouflaged objects. We adopt Res2Net as the backbone encoder. After capturing rich context information by the Residual Multi-scale Feature Extractor (RMFE), we design the Separated Attention (SEA) module to distinguish the subtle difference of foreground and background. The Boundary Guider (BG) module is included in the SEA module to strengthen the model’s ability to understand the boundary. Finally, we employ the Shuffle Attention (SHA) block and a feature fusion module to refine our COD result.

### Result

Here's the experimental result.

![](C:\Users\zhw\Desktop\BSANet\figure\quant-result.png)

![visual-result](C:\Users\zhw\Desktop\BSANet\figure\visual-result.png)

## Usage

### Dependencies

Please refer to requirements.txt

Installing necessary packages: `pip install -r requirements.txt`.

### Datasets

* Train Dataset can be found [here](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view).

* Test Dataset can be found [here](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view).

Once finished, please move the train/test dataset into `./Dataset/`.

### Train

After you download the train dataset, just run `MyTrain.py`. You can change the arguments to customize your preferred training environment settings. The trained model will be saved in `./Snapshot`.

### Test

* BSA-Net uses [Res2Net](https://arxiv.org/abs/1904.01169) as its backbone, so please download Res2Net's pretrained model [here](https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth) and put it into `./Src/Backbone`.

* The BSA-Net pretrained model and prediction maps on 3 benchmark datasets can be found [here](https://drive.google.com/drive/folders/18Z16W73PxZoapJqWmwq-LDfV4NIIqNlG). Please put the pretrained model (`final_35.pth`) into `./Snapshot/`.

- After you download all the pretrained model, just run `MyTest.py` to generate the final prediction map: replace your trained model directory (`--model_path`) and assign your the save directory of the inferred mask (`--test_save`). *(Better not to change `--test_save` since the default path will used by evaluation)*.

### Evaluation

We provide complete and fair one-key evaluation toolbox for benchmarking within a uniform standard. Please refer to this link for more information: 

* Matlab version: https://github.com/DengPingFan/CODToolbox 
* Python version: https://github.com/lartpang/PySODMetrics

Copy the testing GT map (`./Dataset/TestDataset/*/GT`) `./evaluation/GT`, run `./evaluation/evaluation.py`, when the evaluation finished, it will save the metric results into `./result.txt`.

## Contact

If you have any questions, feel free to E-mail me via: `zhuhongwei1999@gmail.com`
