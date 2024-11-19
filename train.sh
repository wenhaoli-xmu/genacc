deepspeed \
    --include localhost:4,5,6,7 \
    train.py \
    --num_layers 32 \
    --max_tokens 4096 \
    --env_conf train/genacc19-14.json \
    --instance_per_cycle 512 \
    --max_prepare_workers 4 \
    --prepare_batch_size_per_gpu 1 \
    --backward_per_head \
    --max_que 1024 \
    --max_top 1024 \
    --max_oth 1024 \
    --beta 3.0 \
    --margin 30
