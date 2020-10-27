#!/bin/sh
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset multi --source painting --target real --num 3 --net resnet34 --pretrained_ckpt ./freezed_models/model_real_sketch_step_3500.pth.tar --save_check

#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset multi --source real --target sketch --num 3 --net resnet34 --pretrained_ckpt ./freezed_models/model_real_sketch_step_3500.pth.tar --save_check

#CUDA_VISIBLE_DEVICES=0 python main.py --method MME --dataset multi --source real --target clipart --num 3 --net resnet34 --pretrained_ckpt ./freezed_models/model_clipart_painting_step_4500.pth.tar --save_check

CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset multi --source real --target painting --num 3 --net resnet34 --pretrained_ckpt ./freezed_models/model_clipart_painting_step_4500.pth.tar --save_check
