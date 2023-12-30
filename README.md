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

--data_path to the physionet dataset directory.

Eg. if path to the physionet dataset is

/Users/parthagrawal02/Desktop/ECG_CNN/physionet/WFDBRecords

then --datapath '/Users/parthagrawal02/Desktop/ECG_CNN/physionet'

For Finetuning :

```
python /kaggle/working/ECG_MAE/main_finetune.py\
    --mode "finetune"
    --model vit_1dcnn \
    --finetune '/checkpoint-360.pth' \
    --epochs 70 \
    --lr 5e-3 \
    --data "PTB" \
    --task "superdiagnostic" \
    --cuda 'CUDA'\
    --warmup_epochs 0 \
    --nb_classes 5 \
    --classf_type "multi_label"

```

Option to select datasets - --data "PTB" for PTB-XL or "physionet" for physionet dataset

If choosing PTB, need to specify --task, and --nb_classes accordingly

For physionet dataset - need to change --classf_type to "multi_class"

For Linear Evaluation

```
python main_finetune.py\
    --model vit_1dcnn \
    --lr 1e-1 \
    --finetune /kaggle/input/check384-large/checkpoint-384.pth \
    --data_path /kaggle/input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ \
    --num_workers 2 \
    --batch_size 32 \
    --accum_iter 1 \
    --cuda "CUDA" \
    --warmup_epochs 0 \
    --nb_classes 5 \
    --epochs 50 \
    --mode "linprobe" \
    --data "PTB" \
    --task "superdiagnostic" \
    --classf_type "multi_label"

```
