# PosFormer

<h3 align="center"> <a href="https://arxiv.org/abs/2407.07764">PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer</a></h3>


<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2407.07764-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.07764)
[![blog](https://img.shields.io/badge/Blog-zhihu-blue)](https://zhuanlan.zhihu.com/p/715387619)
[![poster](https://img.shields.io/badge/poster-blue)](https://github.com/SJTU-DeepVisionLab/PosFormer/blob/main/poster_PosFormer.png)

# Description
 This repository provides the official implementation of the Position Forest Transformer (PosFormer) for Handwritten Mathematical Expression Recognition (HMER). This innovative model introduces a dual-task approach, optimizing both expression and position recognition to facilitate position-aware feature learning for symbols in mathematical expressions. It employs a novel structure called a position forest to parse and model the hierarchical relationships and spatial positioning of symbols without the need for extra annotations. Additionally, an implicit attention correction module is integrated into the sequence-based decoder architecture to enhance focus and accuracy in symbol recognition. PosFormer demonstrates significant improvements over existing methods on several benchmarks, including single-line CROHME datasets and more complex multi-line and nested expression datasets, achieving higher performance without extra computational overhead. This repository includes code, pre-trained models, and usage instructions to aid researchers and developers in applying and further developing this state-of-the-art HMER solution.




## News 
* ```2024.7.10 ``` 🚀 [MNE](https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view?usp=drive_link) available.
* ```2024.7.21 ``` 🚀 Source code and checkpoint (CROHME) available.
* ```2024.8.14 ``` 🚀 Fixed the error in the training command, provided the download link for CROHME, and optimized the README.


## MNE Dataset
The MNE dataset can now be downloaded [here](https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view?usp=drive_link).

The Multi-level Nested Expression (MNE) dataset is specifically designed for evaluating the capability of models to recognize complex handwritten mathematical expressions. It comprises three subsets categorized based on the nested levels of the expressions: N1, N2, and N3, representing one, two, and three levels of nesting, respectively. This dataset avoids using the N4 level due to its minimal representation (only 0.2%) in public datasets. The distribution of the other levels in existing datasets includes 37.4% for N1, 51.4% for N2, 9.7% for N3, and 1.3% for other complexities.

Originally, the images for subsets N1 and N2 were sourced from the CROHME test sets and include 1875 and 304 images, respectively. The N3 subset, containing initially only 10 images, was expanded to 1464 images to provide a robust challenge in identifying highly complex expressions. This expansion was achieved by incorporating complex expression images from public documents and real-world handwritten homework, cited from multiple sources in the research literature. 


## Getting Started

### Installation
```bash
cd PosFormer
# install project   
conda create -y -n PosFormer python=3.7
conda activate PosFormer
conda install pytorch=1.8.1 torchvision=0.2.1 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia 
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
conda install opencv
<conda path>/envs/PosFormer/bin/pip install jsonargparse[signatures] einops
<conda path>/envs/PosFormer/bin/python train.py --config config.yaml
```
### Data Preparation

Please download CROHME and MNE datasets and organize them as follows
```
📂 PosFormer
   ├── 📦 data_crohme.zip
   └── 📦 data_MNE.zip

```
The MNE dataset can now be downloaded [here](https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view?usp=drive_link).

The CROHME dataset can be downloaded [CoMER/blob/master/data.zip](https://github.com/Green-Wood/CoMER/blob/master/data.zip) (provided by the **CoMER** project) 

If you have additional data, you can organize them in a similar way as shown below and compress them into a **zip** file (you can also modify the datamodule to directly input the files).
```
📂 data
   └── 📂 name_of_dataset
       ├── 📂 img
       │   ├── 0.png
       │   ├── 1.png
       │   └── ...
       └── caption.txt
```

### Checkpoint

[model weights](https://github.com/SJTU-DeepVisionLab/PosFormer/tree/main/lightning_logs/version_0/checkpoints)

### Training

For training, we utilize a single A800 GPU; however, an RTX 3090 GPU also provides sufficient memory to support a training batch size of 8. The training process is expected to take approximately 25 hours on a single A800 GPU.

```bash
cd PosFormer
python train.py --config config.yaml
```

### Evaluation 


```bash
cd PosFormer
# results will be printed in the screen and saved to lightning_logs/version_0 folder
bash eval_all_crohme.sh 0
```

 ### TODO
 1. update LICENSE file 
 2. Improve README and samples

### Citation
```
@article{guan2024posformer,
  title={PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer},
  author={Guan, Tongkun and Lin, Chengyu and Shen, Wei and Yang, Xiaokang},
  journal={arXiv preprint arXiv:2407.07764},
  year={2024}
}
```

### License
```
- This code is only free for academic research purposes and licensed under the 2-clause BSD License. Parts of this project contain code from other sources, which are subject to their respective licenses.
```
