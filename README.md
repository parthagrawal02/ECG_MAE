# ECG_MAE

Implementation of Spatiotemporal self-supervised representation learning from multi-lead ECG signals
(https://www.sciencedirect.com/science/article/pii/S1746809423002057#bib1) using https://github.com/facebookresearch/mae.

To Pretrain run :

```
python main_pretrain.py \
    --batch_size 64 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 10 \
    --data_path ${IMAGENET_DIR} \
    --lr 1e-3 \
    --cuda "CUDA"

```

data_path to the physionet -

Eg. if path to the physionet dataset is 

/Users/parthagrawal02/Desktop/ECG_CNN/physionet/WFDBRecords

then --datapath '/Users/parthagrawal02/Desktop/ECG_CNN/physionet'

To Finetune :

```
!python /kaggle/working/ECG_MAE/main_finetune.py\
    --model vit_1dcnn \
    --finetune '/checkpoint-360.pth' \
    --epochs 70 \
    --lr 5e-3 \
    --data_path /Users/parthagrawal02/Desktop/ECG_CNN/physionet \
    --cuda 'CUDA'\
    --val_start 37 --val_end 41 --train_start 0 --train_end 37
```

Change ecg_dataloader according to the dataset
