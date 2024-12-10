train_script=train/llama3-8b-spotlight.json

deepspeed \
    --include localhost:3,4,5,6 \
    train.py \
    --num_layers 32 \
    --max_tokens 512 \
    --env_conf $train_script \
    --instance_per_cycle 2048 \
    --max_prepare_workers 16 \
    --prepare_batch_size_per_gpu 4 \
    --max_que 1024 \
    --max_top 1024 \
    --max_oth 1024 \
    --beta 1.0 \
    --margin 0.0 \
    --maskout 0.98 \
    --use_prepared_data 

python train_results/convert.py --env_conf $train_script