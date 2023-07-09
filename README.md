# Layout Generation as Intermediate Action Sequence Prediction - Official Pytorch Impletation

[Paper ](https://ojs.aaai.org/index.php/AAAI/article/view/26277) | [Supplement](https://github.com/BERYLSHEEP/LayoutActionProject/blob/main/4740-supp.pdf)

Layout Generation as Intermediate Action Sequence Prediction 

Huiting Yang  , Danqing Huang  , Chin-Yew Lin  , Shengfeng He 

*In AAAI 2023 Oral*



## Prerequisites

- Linux

- Python3

- NVIDIA GPU + CUDA CuDNN

  

## Environment

Start a new conda environment

```
conda env create -f requirements.yml
conda activate layout
```

## Resources

The resources related to Infoppt will be released publicly after the original author open sources them.

Different SOTA pre-trained models and datasets are saved in[ `Resources` ](https://drive.google.com/drive/folders/1KU9q83gzKD2HGoBduN2CWC0LHUmDcFy0?usp=drive_link)

```dircolors
    Resources/
    └── pretrained_model_resources
        └── LayoutTransformer
        └── LayoutGAN++
        └── LayoutVAE
        └── Ours
            ├── rico.pth
            ├── publaynet.pth
            ├── infoppt.pth
    └── processed_data
        └── LayoutGAN++
            └──infoppt
               └──processed
                  ├── pre_filter.pt
            			├── pre_transform.pt
            			├── test.pt
            			├── val.pt
            └──publaynet
            └──rico 
```

### Data preparation

#### Create a folder for datasets

```
TOPDIR=$(git rev-parse --show-toplevel)
DATASET=$TOPDIR/LayoutAction/datasets
mkdir $DATASET
```

#### Download and place the datasets

[Rico](http://www.interactionmining.org/rico.html)

- Download `rico_dataset_v0.1_semantic_annotations.zip` from "UI Screenshots and Hierarchies with Semantic Annotations" and decompress it.

- Create the new directory `$DATASET/rico/` and move the contents into it as shown below:

```dircolors
    $DATASET/rico/
    └── semantic_annotations
        ├── 0.json
        ├── 0.png
        ├── 10000.json
        ├── 10000.png
        ├── 10002.json
        ├── ...
```

[PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/)

- Download `labels.tar.gz` and decompress it.

- Create the new directory `$DATASET/publaynet/` and move the contents into it as shown below:

  ```dircolors
  $DATASET/
  └── publaynet
      ├── LICENSE.txt
      ├── README.txt
      ├── train.json
      └── val.json
  ```

[InfoPPT]()

- Create the new directory `$DATASET/infoppt/` and move the contents into it as shown below:

  ```dircolors
  $DATASET/
  └── infoppt
      ├── 1.pptx
      ├── 2.pptx
      ├── 3.pptx
      └── ...
  ```

## Training

```cmd
cd LayoutAction
python main.py --exp test --dataset publaynet --device 0 --log_dir ./output/logs
```

Results are saved in output/logs/publaynet/test. Dataset options: ['rico', 'publaynet', 'infoppt']



## Evaluation

First, obtain the .pkl file of containing the bounding boxes and labels.

```cmd
cd LayoutAction
python main.py --exp test --dataset publaynet --device 0 --evaluate --model_path ../Resources/pretrained_model_resources/Ours/publaynet.pth --eval_command category_generate --save_pkl
```

Here, we obtain the category conditional generated file in `LayoutAction/output/logs/publaynet/test/generated_layout.pth`

### FID Evaluation

This ablation study is only excuated in Rico dataset. 

For more detailed comparative experiments, please refer to this document "FID_disc/README.md".

#### Pre-training the FID network

To unify the training data of each model, we first [download](https://www.kaggle.com/datasets/onurgunes1993/rico-dataset) and divide the rico dataset into three splits: train, validation, and test. The original rico dataset should be placed in "LayoutAction/datasets/rico".

```cmd
cd FID_disc
python divide_rico.py --rico_dir ../LayoutAction/datasets/rico/semantic_annotations/ --out_dir ./data/rico
```

Move the processed dataset to the correct place

```cmd
cd FID_disc/LayoutGAN++
mkdir datas/dataset/rico/raw
mv Resources/processed_data/LayoutGAN++/rico/processed \
		FID_disc/LayoutGAN++/datas/dataset/rico
```

To obtain the real image dataset.

```cmd
python shuffle_gauss_eval.py --dataset rico --split train --command real --save_img \
        --out_dir ../PixelInput/output/rico/train_real
python shuffle_gauss_eval.py --dataset rico --split val --command real --save_img \
        --out_dir ../PixelInput/output/rico/val_real
python shuffle_gauss_eval.py --dataset rico --split test --command real --save_img \
        --out_dir ../PixelInput/output/rico/test_real
```

To obtain the fake image dataset which negative cases contain in-batch shuffle and gauss images.

```cmd
python shuffle_gauss_eval.py --dataset rico --split train --command shuffle_gauss --save_img  --out_dir ../PixelInput/output/rico/train_fake
python shuffle_gauss_eval.py --dataset rico --split val --command shuffle_gauss --save_img  --out_dir ../PixelInput/output/rico/train_val
python shuffle_gauss_eval.py --dataset rico --split test --command shuffle_gauss --save_img  --out_dir ../PixelInput/output/rico/test_fake
```

To train the FID network

```cmd
cd FID_disc/PixelInput
python train_FIDnet.py train \
    --real_img_dir ./output/rico/train_real --false_img_dir ./output/rico/train_fake \
    --real_val_img_dir ./output/rico/val_real --false_val_img_dir ./output/rico/val_fake
```

To calculate the  accuracy

```cmd
python train_FIDnet.py cal_accuracy \
    --real_img_dirs ./output/rico/test_real --false_img_dir ./output/rico/test_fake \
    --model_path "pre-trained model path" 
```

#### FID evaluation

```
cd FID_disc/LayoutGAN++
python shuffle_gauss_eval.py --dataset rico --split test --command real --save_img \
        --out_dir ../PixelInput/output/rico/test_real

cd FID_disc/PixelInput
python train_FIDnet.py eval --dataset rico --pkl_path ../../LayoutAction/output/logs/rico/test/generated_layout.pth --save_img_dir ./output/publaynet/ours/test --real_img_dir ./output/rico/test_real --model_path ../pretrained_model/rico_resnet.pth.tar
```

- `--pkl_path`: .pkl is the generated file obtained from previous step
- `--save_img_dir`: the layouts corresponding to the .pkl file are saved under this directory
- `--real_img_dir`: this directory is the same as the `{args.out_dir}` in the above command
- `--model_path`: the pre-trained model path 

### Heuristic metrics evaluation (IoU, Align and Overlap)

We reuse the heuristic metrics from [LayoutGAN++](https://github.com/ktrk115/const_layout). 

- Prepare the dataset for LayoutGAN++, see [this instruction](https://github.com/ktrk115/const_layout/tree/master/data) or move the preprocess data to the corresponding directory.

```
mv Resources/processed_data/LayoutGAN++/infoppt const_layout/datas/dataset
mkdir datas/dataset/infoppt/raw
mv Resources/processed_data/LayoutGAN++/publaynet const_layout/datas/dataset
mkdir datas/dataset/publaynet/raw
mv Resources/processed_data/LayoutGAN++/rico const_layout/datas/dataset
mkdir datas/dataset/rico/raw
```

- Excuate the following command:

```
cd const_layout
python eval.py --dataset publaynet --pkl_paths ../LayoutAction/output/logs/publaynet/test/generated_layout.pth  --compute_real
```

