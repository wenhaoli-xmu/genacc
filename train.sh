deepspeed \
    --include localhost:5,6 \
    train.py \
    --num_layers 32 \
    --max_tokens 8192 \
    --env_conf train/genacc19-12.json \
    --instance_per_cycle 2000 \
    --prepare_batch_size_per_gpu 1 \
    --max_que 256 \
    --max_oth 256 \
    --max_top 256 \
    --maskout 0.98
