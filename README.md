# Robust Visual Grounding with Transformers

This is our project for the class Introduction to Deep Learning, 11-785, in CMU. The task is to extract textual features from the input query and localize the corresponding object(s) in the image. We approach this problem by using a transformer based architecture called VGTR, which is based on the paper, Visual Grounding with Transformers. This repository is based on top of the VGTR repository and more information about that is provided below.

We propose new 2 different attention mechanisms namely, Early Attention and Alternating Co-Attention to enhance the cross-modal fusion between visual and textual features. In addition, we use Contrastive Learning to help the model learn better connections between positive image-query pairs. We also replace the previous textual backbone -- Bi-LSTM -- with BERT. 

The repository is divided into 3 branches --
- BERT - Contains the code for training the model with BERT as textual backbone.
- Contrastive loss - Contains the code for Contrastive Learning and Alternating Co-Attention.
- Early Attention - Contains the code for Early Dot Attention and Early Co-Attention. 

The model is trained on the RefCOCO dataset and tested on RefCOCO and RefCOCO+ testA and testB sets. The final accuracy of different proposed methods is shown in the table below. Our proposed method combining Contrastive Learning and Alternating Co-Attention performs really well and also out-performs the baseline framework in RefCOCO and RefCOCO+ testA sets.

Full report available here - [[Report](https://drive.google.com/file/d/1T3EPAlZVEBu-3TAwPuCr74VxmkIPaeSV/view?usp=sharing)]

![Alt text](result.PNG?raw=true "Title")

Contributors - 
- Shreyas Misra
- Nitika Suresh
- Jigarkumar Patel
- Gunjan Sethi

##  Overview

This repository includes PyTorch implementation and trained models of VGTR(**V**isual **G**rounding with **TR**ansformers).

[[arXiv](https://arxiv.org/abs/2105.04281)]


>In this paper, we propose a transformer based approach for visual grounding. Unlike existing proposal-and-rank frameworks that rely heavily on pretrained object detectors or proposal-free frameworks that upgrade an off-the-shelf one-stage detector by fusing textual embeddings, our approach is built on top of a transformer encoder-decoder and is independent of any pretrained detectors or word embedding models. Termed as VGTR – Visual Grounding with TRansformers, our approach is designed to learn semantic-discriminative visual features under the guidance of the textual description without harming their location ability. This information flow enables our VGTR to have a strong capability in capturing context-level semantics of both vision and language modalities, rendering us to aggregate accurate visual clues implied by the description to locate the interested object instance. Experiments show that our method outperforms state-of-the-art proposal-free approaches by a considerable margin on four benchmarks.

<img width="805" alt="图片" src="https://user-images.githubusercontent.com/83934424/157177788-534d16e8-c91c-432d-8939-213c7f3065a2.png">


## Prerequisites

- python 3.6
- pytorch>=1.6.0
- torchvision
- CUDA>=9.0
- others (opencv-python etc.)


## Preparation

   1. Clone this repository.

   2. Data preparation.

      Download Flickr30K Entities from [Flickr30k Entities (bryanplummer.com)](http://bryanplummer.com/Flickr30kEntities/) and  [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/) 

      Download MSCOCO images from [MSCOCO](http://images.cocodataset.org/zips/train2014.zip)

      Download processed indexes from [Gdrive](https://drive.google.com/drive/folders/1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ?usp=drive_open), process by [zyang-ur
   ](https://github.com/zyang-ur/onestage_grounding).

   3. Download backbone weights. We use resnet-50/101 as the basic visual encoder. The weights are pretrained on MSCOCO, and can be downloaded here (BaiduDrive):

      [ResNet-50](https://pan.baidu.com/s/1ZHR_Ew8tUZH7gZo1prJThQ)(code：ru8v);  [ResNet-101](https://pan.baidu.com/s/1zsQ67cUZQ88n43-nmEjgvA)(code：0hgu).

   4. Organize all files like this：

   ```bash
   .
   ├── main.py
   ├── store
   │   ├── data
   │   │   ├── flickr
   │   │   │   ├── corpus.pth
   │   │   │   └── flickr_train.pth
   │   │   ├── gref
   │   │   └── gref_umd
   │   ├── ln_data
   │   │   ├── Flickr30k
   │   │   │   └── flickr30k-images
   │   │   └── other
   │   │       └── images
   │   ├── pretrained
   │   │   └── flickr_R50.pth.tar
   │   └── pth
   │       └── resnet50_detr.pth
   └── work
   ```


## Model Zoo

   | Dataset           | Backbone  | Accuracy            | Pretrained Model (BaiduDrive)                                |
   | ----------------- | --------- | ------------------- | ------------------------------------------------------------ |
   | Flickr30K Entites | Resnet50  | 74.17               | [flickr_R50.pth.tar](https://pan.baidu.com/s/1VUnxD-5pXnM7iFwIl8q9kA) code: rpdr |
   | Flickr30K Entites | Resnet101 | 75.32               | [flickr_R101.pth.tar](https://pan.baidu.com/s/10GcUFLSTei9Lwvu4e5GjrQ) code: 1igb |
   | RefCOCO           | Resnet50  | 78.70  82.09  73.31 | [refcoco_R50.pth.tar](https://pan.baidu.com/s/1GIe5OoOQOADYc1vVGcSXbw) code: xjs8 |
   | RefCOCO           | Resnet101 | 79.30  82.16  74.38 | [refcoco_R101.pth.tar](https://pan.baidu.com/s/1GL-itH93G_e3VVNUPtocSA) code: bv0z |
   | RefCOCO+          | Resnet50  | 63.57  69.65  55.33 | [refcoco+_R50.pth.tar](https://pan.baidu.com/s/1PUF8WoTrOLmYU24kgAMXKQ) code: 521n |
   | RefCOCO+          | Resnet101 | 64.40  70.85  55.84 | [refcoco+_R101.pth.tar](https://pan.baidu.com/s/1mJiA7i7-Mp5ZL5D6dEDy0g) code: vzld |
   | RefCOCOg          | Resnet50  | 62.88               | [refcocog_R50.pth.tar](https://pan.baidu.com/s/1KvDPisgSLzy8u5bIVCBiOg) code: wb3x |
   | RefCOCOg          | Resnet101 | 64.05               | [refcocog_R101.pth.tar](https://pan.baidu.com/s/13ubLIbIUA3XlhzSOjaK7dg) code: 5ok2 |
   | RefCOCOg-umd      | Resnet50  | 65.62  65.30        | [umd_R50.pth.tar](https://pan.baidu.com/s/1-PgzbA98rUOl7VJHAO-Exw) code: 9lzr |
   | RefCOCOg-umd      | Resnet101 | 66.83  67.28        | [umd_R101.pth.tar](https://pan.baidu.com/s/1JkGbYL8Of3WOVWI9QcVwhQ) code: zen0 |


## Train

   ```
   python main.py \
      --gpu $gpu_id \
      --dataset $[refcoco | refcoco+ | others] \
      --batch_size $bs \
      --savename $exp_name \
      --backbone $[resnet50 | resnet101] \
      --cnn_path $resnet_coco_weight_path
   ```
   

## Inference

   Download the pretrained models and put it into the folder ```./store/pretrained/```.

   ```
   python main.py \
      --test \
      --gpu $gpu_id \
      --dataset $[refcoco | refcoco+ | others] \
      --batch_size $bs \
      --pretrain $pretrained_weight_path
   ```

## Acknowledgements

Part of codes are from:

   1. [facebookresearch/detr](https://github.com/facebookresearch/detr)；
   2. [zyang-ur/onestage_grounding](https://github.com/zyang-ur/onestage_grounding)； 
   3. [andfoy/refer](https://github.com/andfoy/refer)；
   4. [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).


 
## Citation
   ```
   @article{du2021visual,
     title={Visual grounding with transformers},
     author={Du, Ye and Fu, Zehua and Liu, Qingjie and Wang, Yunhong},
     journal={arXiv preprint arXiv:2105.04281},
     year={2021}
   }
   ```
   
