deepspeed \
    --include localhost:0,1,2,3 \
    train_tuning.py \
    --fix_layer 2 \
    --max_tokens 8192 \
    --env_conf train/genacc19-13.json \
    --instance_per_cycle 4000 \
    --max_prepare_workers 4 \
    --prepare_batch_size_per_gpu 1 \
    --max_que 256 \
    --max_oth 256 \
    --max_top 256 \
    --maskout 0.98 \
    --lr "[0.1,0.01,0.001,0.0001]" \
    --beta "[0.3,1,3]" \
    --margin "[0.3,1,3]"
