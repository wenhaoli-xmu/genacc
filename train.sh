deepspeed \
    --include localhost:0,1,2,3 \
    train.py \
    --num_layers 32 \
    --max_tokens 4096 \
    --env_conf train/genacc19-14.json \
    --instance_per_cycle 400 \
    --max_prepare_workers 4 \
    --prepare_batch_size_per_gpu 1 \
    --max_que 256 \
    --max_oth 256 \
    --max_top 256 \
    --beta 3.0 \
    --margin 30
