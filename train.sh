deepspeed \
    --include localhost:0,1,2,3 \
    train.py \
    --num_layers 32 \
    --max_tokens 8192 \
    --env_conf train/genacc19-13.json \
    --instance_per_cycle 2000 \
    --max_prepare_workers 4 \
    --prepare_batch_size_per_gpu 1 \
    --max_que 256 \
    --max_oth 256 \
    --max_top 256
