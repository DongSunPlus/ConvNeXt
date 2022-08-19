#!/usr/bin/env bash
###
 # @Author: DongSunPlus dong.sun@plus.ai
 # @Date: 2022-07-23 21:44:44
 # @LastEditors: DongSunPlus dong.sun@plus.ai
 # @LastEditTime: 2022-08-19 21:24:38
 # @FilePath: /ConvNeXt/exp.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# train
exp_id=plusai_$(date "+%Y%m%d-%H%M%S")

echo "$exp_id"

train() {

    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
    array=(`echo $CUDA_VISIBLE_DEVICES | tr ',' ' '`)
    nums_node=${#array[@]}
    python -m torch.distributed.launch --nproc_per_node=$nums_node main.py \
    --model convnext_tiny \
    --finetune /mnt/jupyterhub/dong.sun/workspace/ConvNeXt/weights/convnext_tiny_22k_224.pth\
    --model_prefix /mnt/jupyterhub/dong.sun/workspace/ConvNeXt/weights/convnext_tiny_22k_224.pth\
    --drop_path 0.1 \
    --batch_size 60 \
    --input_size 224 \
    --data_set image_folder\
    --nb_classes 7 \
    --gpus_str $CUDA_VISIBLE_DEVICES \
    --lr 4e-3 \
    --update_freq 4 \
    --eval false \
    --model_ema true \
    --model_ema_eval true \
    --data_path /mnt/jupyterhub/dong.sun/data/iqa_datav2/train \
    --eval_data_path /mnt/jupyterhub/dong.sun/data/iqa_datav2/val \
    --output_dir /mnt/jupyterhub/dong.sun/workspace/ConvNeXt/exp/$exp_id/model \
    --log_dir /mnt/jupyterhub/dong.sun/workspace/ConvNeXt/exp/$exp_id/log
}

eval() {

    export CUDA_VISIBLE_DEVICES=1
    array=(`echo $CUDA_VISIBLE_DEVICES | tr ',' ' '`)
    nums_node=${#array[@]}
    python -m torch.distributed.launch --nproc_per_node=$nums_node main.py \
    --model convnext_tiny \
    --eval true \
    --data_set image_folder \
    --resume /home/dong.sun/git/ConvNeXt/exp/plusai_20220809-154016/model/checkpoint-best.pth \
    --input_size 448 \
    --drop_path 0.1 \
    --nb_classes 7 \
    --gpus_str $CUDA_VISIBLE_DEVICES \
    --data_path /home/dong.sun/git/ConvNeXt/data/datav2/train \
    --eval_data_path /home/dong.sun/git/ConvNeXt/data/datav2/val
}
   
if [ "$1" = "train" ]; then
    train
elif [ "$1" = "eval" ]; then
    eval  
fi
