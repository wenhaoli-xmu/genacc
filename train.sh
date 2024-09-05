#!/bin/bash


for i in range(0, 4):

    handles = []

    for j in range(0, 8):
        layer_idx = i * 8 + j

        handle = execute(CUDA_VISIBLE_DEVICES=j python train/train_layer_wise.py \
            --env_conf train/genacc19-6.json \
            --layer layer_idx \
            --max_oth 1024)

        handles.append(handle)

    for handle in handles:
        handle.join()