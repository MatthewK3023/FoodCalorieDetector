#!/usr/bin/env sh
export GITROOT=$(python test.py \
--task ctdet \
--gpus 0 \
--dataset food100 \
--load_model ../exp/ctdet/test_food100_fpn_2/model_last.pth \
--trainval \
--arch mobilenetv2fpn_10 \
)
