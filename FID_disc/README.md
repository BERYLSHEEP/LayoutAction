## Discriminator for evaluating FID scores:
This ablation study is only excuated in Rico dataset.

To unify the training data of each model, we first divide the rico dataset into three splits: train, validation, and test.
```
python divide_rico.py --rico_dir ../LayoutAction/datasets/rico/semantic_annotations/ --out_dir ./data/rico
```

- Transformer Encoder, which is the same as the LayoutTransformer backbone
```
cd Transformer
# Training
python train_FIDnet.py --command train \
    --train_dir ./data/rico/train --val_dir ./data/rico/val
                
#Calculate accuracy
python train_FIDnet.py --command cal_acc \
    --test_dir ./data/rico/test --model_path "pre-trained model path"
```

- Transformer Encoder-Decoder, which is the same as LayoutGAN++
```
cd LayoutGAN++
# Train
    python train_FIDnet.py --command train --negative_case gauss_shuffle \
        --train_dir ./data/rico/train --val_dir ./data/rico/val
        
# --negative_case option: "gauss_shuffle" or "only_shuffle"

# Calculate accuracy
    python train_FIDnet.py --command cal_acc --negative_case gauss_shuffle \
        --test_dir ./data/rico/test
```

- Embedding input + Transformer Encoder-Decoder, which dose not show in the paper
```
cd Transformer 
# Train
python train_FIDnet.py --command train --is_enc_dec \
    --train_dir ./data/rico/train --val_dir ./data/rico/val
        
# Calculate accuracy
python train_FIDnet.py --command cal_acc --is_enc_dec \
    --test_dir ./data/rico/test --model_path "pre-trained model path"
```    

- ResNet-18, a CNN-based model which takes the layout image as input.
```
cd FID_disc/LayoutGAN++
# obtain the real image dataset
python shuffle_gauss_eval.py --dataset rico --split train --command real --save_img \
        --out_dir ../PixelInput/output/rico/train_real
python shuffle_gauss_eval.py --dataset rico --split test --command real --save_img \
        --out_dir ../PixelInput/output/rico/test_real

# obtain the fake image dataset which negative cases contain in-batch shuffle and gauss.
python shuffle_gauss_eval.py --dataset rico --split train --command shuffle_gauss --save_img \
        --out_dir ../PixelInput/output/rico/train_fake
python shuffle_gauss_eval.py --dataset rico --split test --command shuffle_gauss --save_img \
        --out_dir ../PixelInput/output/rico/test_fake

cd FID_disc/PixelInput
# Train
python train_FIDnet.py train \
    --real_img_dataset ./output/rico/train_real --false_img_dataset ./output/rico/train_fake

# Calculate accuracy
python train_FIDnet.py cal_accuracy \
    --real_img_dataset ./output/rico/test_real --false_img_dataset ./output/rico/test_fake \
    --model_path "pre-trained model path" 
```

- Layout Matching Network (GMN), which uses the graph structure to represent a layout for layout similarity learning
```
cd LayoutGMN
# Train
python train_FIDnet.py --train_mode \
    --train_dir ./data/rico/train --val_dir ./data/rico/val --model_save_name "checkpoint_D.pth"

# Calculate accuracy
python train_FIDnet.py --model_path "pre-trained model path" --test_dir ./data/rico/test
```

## Generating the image for evaluating FID scores with Resnet-18 discriminator
-  `cd LayoutGAN++`
- Save the shuffle image in the directory `../PixelInput/output/rico/shuffle`
```
python shuffle_gauss_eval.py --dataset rico --split test --command shuffle --save_img --out_dir ../PixelInput/output/rico/shuffle
```
- Save the gauss image and its corresponding pkl in the directory `../PixelInput/output/rico/gauss`
```
python shuffle_gauss_eval.py --dataset rico --split test --command gauss --save_img --save_pkl --out_dir ../PixelInput/output/rico/gauss
```
- other commands include "save real image", "save miss_elements image", "save shift image", "shuffle_gauss".

## Resources
- Generated rico layouts for evaluating FID scores are saved in `FID_disc/rico_generated_layouts`

- Pre-trained discriminator models are saved in `FID_disc/pretrained_model`