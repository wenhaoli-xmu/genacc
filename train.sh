deepspeed --include localhost:2,3,4,5 train.py \
    --num_layers 32 \
    --max_tokens 4096 \
    --env_conf train/genacc19-11.json \
    --instance_per_cycle 500 \
    --max_oth 512 \
    --maskout 0.98