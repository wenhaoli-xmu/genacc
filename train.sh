deepspeed \
    --include localhost:0,1,2,3 \
    train.py \
    --num_layers 32 \
    --max_tokens 4096 \
    --env_conf train/genacc23-1.json \
    --instance_per_cycle 1024 \
    --max_prepare_workers 4 \
    --prepare_batch_size_per_gpu 1 \
    --backward_per_head \
    --max_que 1024 \
    --max_top 1024 \
    --max_oth 1024 \
    --beta 1.0 \
    --margin 0
