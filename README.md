# ECG_MAE

Implementation of Spatiotemporal self-supervised representation learning from multi-lead ECG signals
(https://www.sciencedirect.com/science/article/pii/S1746809423002057#bib1) using https://github.com/facebookresearch/mae.

To Pretrain run :

```
python main_pretrain.py \
    --batch_size 64 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --data_path ${IMAGENET_DIR}
```

data_path to the physionet - 

Eg. if path to the physionet dataset is /Users/parthagrawal02/Desktop/ECG_CNN/physionet/WFDBRecords

then --datapath '/Users/parthagrawal02/Desktop/ECG_CNN'
