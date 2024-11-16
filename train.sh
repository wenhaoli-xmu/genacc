deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    train.py \
    --num_layers 32 \
    --max_tokens 4096 \
    --env_conf train/genacc19-14.json \
    --instance_per_cycle 400 \
    --max_prepare_workers 4 \
    --prepare_batch_size_per_gpu 1 \
    --max_que 512 \
    --max_oth 512 \
    --max_top 512 \
    --beta 3.0 \
    --margin 30
