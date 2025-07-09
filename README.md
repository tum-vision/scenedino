<div align="center">
<h1>Feed-Forward <i>SceneDINO</i> for Unsupervised Semantic Scene Completion</h1>


[**Aleksandar Jevtić**](https://jev-aleks.github.io/)<sup>* 1</sup>
[**Christoph Reich**](https://christophreich1996.github.io/)<sup>* 1,2,4,5</sup>
[**Felix Wimbauer**](https://fwmb.github.io/)<sup>1,4</sup>
[**Oliver Hahn**](https://olvrhhn.github.io/)<sup>2</sup>
[**Christian Rupprecht**](https://chrirupp.github.io/)<sup>3</sup>
[**Stefan Roth**](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<sup>2,5,6</sup>
[**Daniel Cremers**](https://cvg.cit.tum.de/members/cremers/)<sup>1,4,5</sup>


<sup>1</sup>TU Munich   <sup>2</sup>TU Darmstadt   <sup>3</sup>University of Oxford   <sup>4</sup>MCML   <sup>5</sup>ELIZA   <sup>6</sup>hessian.AI   *equal contribution
<h3>ICCV 2025</h3>


<a href="https://arxiv.org/abs/2507.06230"><img src='https://img.shields.io/badge/ArXiv-grey' alt='Paper PDF'></a>
<a href="https://visinf.github.io/scenedino/"><img src='https://img.shields.io/badge/Project Page-grey' alt='Project Page URL'></a>
<a href="https://huggingface.co/spaces/jev-aleks/SceneDINO"><img src='https://img.shields.io/badge/🤗 Demo-grey' alt='Project Page URL'></a>

<a href="https://opensource.org/licenses/Apache-2.0"><img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'></a>
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)


<center>
    <img src="./assets/scenedino.gif" width="512">
</center>
</div>
<br>

**TL;DR:** SceneDINO is unsupervised and infers 3D geometry and features from a single image in a feed-forward manner. Distilling and clustering SceneDINO's 3D feature field results in unsupervised semantic scene completion predictions. SceneDINO is trained using multi-view self-supervision.

## Abstract

Semantic scene completion (SSC) aims to infer both the 3D geometry and semantics of a scene from single images. In contrast to prior work on SSC that heavily relies on expensive ground-truth annotations, we approach SSC in an unsupervised setting. Our novel method, SceneDINO, adapts techniques from self-supervised representation learning and 2D unsupervised scene understanding to SSC. Our training exclusively utilizes multi-view consistency self-supervision without any form of semantic or geometric ground truth. Given a single input image, SceneDINO infers the 3D geometry and expressive 3D DINO features in a feed-forward manner. Through a novel 3D feature distillation approach, we obtain unsupervised 3D semantics. In both 3D and 2D unsupervised scene understanding, SceneDINO reaches state-of-the-art segmentation accuracy. Linear probing our 3D features matches the segmentation accuracy of a current supervised SSC approach. Additionally, we showcase the domain generalization and multi-view consistency of SceneDINO, taking the first steps towards a strong foundation for single image 3D scene understanding.

## News

- `09/07/2025`: [ArXiv](https://arxiv.org/abs/2507.06230) preprint and code released. 🚀

## Setup (Installation & Datasets)

### Python Environment

Our Python environment is managed with **Conda**.

```shell
conda env create -f environment.yml
conda activate scenedino
```

### Datasets

We provide configuration files for the datasets SceneDINO is trained and evaluated on. Adjust these files and, most importantly, insert the data paths you use.

```bash
configs/dataset/kitti_360_sscbench.yaml
configs/dataset/cityscapes_seg.yaml
configs/dataset/bdd_seg.yaml
configs/dataset/realestate10k.yaml
```

#### KITTI-360

To download KITTI-360, create and account and follow the instructions on the [official website](https://www.cvlibs.net/datasets/kitti-360/index.php). We require the perspective images, fisheye images, raw velodyne scans, calibrations, and vehicle poses.

### Checkpoints

Our pre-trained checkpoints are stored in the CVG webshare. Download one of the checkpoints using the dedicated script. To replicate our results using ORB-SLAM3, we provide the obtained poses in `datasets/kitti_360/orb_slam_poses`.

```bash
# Download best model trained on KITTI-360 (SSCBench split)
python download_checkpoint.py ssc-kitti-360-dino
python download_checkpoint.py ssc-kitti-360-dino-orb-slam
python download_checkpoint.py ssc-kitti-360-dinov2
```

**Table 1. SSCBench-KITTI-360 results.** We compare SceneDINO to the STEGO + S4C baseline in unsupervised SSC using the mean intersection over union score (mIoU) in %.
<table><thead>
  <tr>
    <th>Method</th>
    <th>Checkpoint</th>
    <th colspan="3">mIoU</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td><em>12.8m</em></td>
    <td><em>25.6m</em></td>
    <td><em>51.2m</em></td>
  </tr>
  <tr>
    <td>Baseline</td>
    <td>-</td>
    <td>10.53</td>
    <td>9.26</td>
    <td>6.60</td>
  </tr>
  <tr>
    <td>SceneDINO</td>
    <td><a href="https://huggingface.co/jev-aleks/SceneDINO/tree/main/seg-best-dino">ssc-kitti-360-dino</a></td>
    <td>10.76</td>
    <td>10.01</td>
    <td>8.00</td>
  </tr>
  <tr>
    <td>SceneDINO (ORB-SLAM3 poses)</td>
    <td><a href="https://huggingface.co/jev-aleks/SceneDINO/tree/main/seg-best-dino-orb-slam">ssc-kitti-360-dino-orb-slam</a></td>
    <td>10.88</td>
    <td>9.86</td>
    <td>7.88</td>
  </tr>
  <tr>
    <td>SceneDINO (DINOv2)</td>
    <td><a href="https://huggingface.co/jev-aleks/SceneDINO/tree/main/seg-best-dinov2">ssc-kitti-360-dinov2</a></td>
    <td>13.76</td>
    <td>11.78</td>
    <td>9.08</td>
  </tr>
</tbody>
</table>

## Inference Demo Script

This simple demo script demonstrates loading a model and performing inference in 3D and rendered 2D. It can be used as a starting point to experiment with SceneDINO feature fields.

```bash
python demo_script.py -h

# First image of kitti-360 test set
python demo_script.py --ckpt <PATH-MODEL-CKPT>
# Custom image
python demo_script.py --ckpt <PATH-MODEL-CKPT> --image <PATH-DEMO-IMAGE>
```

## Training

For unsupervised SSC, training is performed in two stages. We provide training configurations in ```configs/``` for each of them. 

**SceneDINO**

First, the 3D feature fields of SceneDINO are trained. 

```bash
python train.py -cn train_scenedino_kitti_360
```

**Unsupervised SSC**

Based on a SceneDINO checkpoint, we train the unsupervised SSC head.

```bash
python train.py -cn train_semantic_kitti_360
```

**Logging**

We use TensorBoard to keep track of losses, metrics, and qualitative results.

```bash
tensorboard --port 8000 --logdir out/
```

## Evaluation

We further provide configurations to reproduce the evaluation results from the paper.

**Unsupervised 2D Segmentation**

```bash
# Unsupervised 2D Segmentation
python eval.py -cn evaluate_semantic_kitti_360
```

**Unsupervised SSC**

```bash
# Unsupervised SSC, adapted from S4C (https://github.com/ahayler/s4c)
python evaluate_model_sscbench.py -ssc <PATH-SSCBENCH> -vgt <PATH-SSCBENCH-LABELS> -cp <PATH-CHECKPOINT>.pt -f -m scenedino -p <RUN-NAME>
```

## Citation

If you find our work useful, please consider citing our paper.
```
@inproceedings{Jevtic:2025:SceneDINO,
    author  = {Aleksandar Jevti{\'c} and
               Christoph Reich and
               Felix Wimbauer and
               Oliver Hahn and
               Christian Rupprecht and
               Stefan Roth and
               Daniel Cremers},
    title   = {Feed-Forward {SceneDINO} for Unsupervised Semantic Scene Completion},
    journal = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    year    = {2025},
}
```

## Acknowledgements

This repository is based on the [Behind The Scenes (BTS)](https://github.com/Brummi/BehindTheScenes) code base.
