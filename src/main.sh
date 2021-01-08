#!/usr/bin/env sh
export GITROOT=$(python main.py \
--task ctdet \
--gpus 4 \
--dataset food100 \
--exp_id food100_fpn_newimagenet_bs16/ \
--reg_loss l1  \
--arch mobilenetv2fpn_10 \
--batch_size 16 \
)
