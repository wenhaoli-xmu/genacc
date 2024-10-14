deepspeed --include localhost:0,1,2,3 train.py \
    --num_layers 32 \
    --env_conf train/genacc19-10.json \
    --instance_per_cycle 10 \
    --max_oth 512 \
    --maskout 0.98