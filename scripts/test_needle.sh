
# test llama-2-7b
python test_needle/run_needle_in_haystack.py \
    --env_conf test_needle/llama2-7b.json \
    --chat_template llama-2 \
    --s_len 128 \
    --e_len 4096 \
    --step 128

python test_needle/viz.py \
    --env_conf test_needle/llama2-7b.json \

python test_needle/run_needle_in_haystack.py \
    --env_conf test_needle/maskout_0.98.json \
    --chat_template llama-2 \
    --s_len 128 \
    --e_len 4096 \
    --step 128

python test_needle/viz.py \
    --env_conf test_needle/maskout_0.98.json \


python test_needle/run_needle_in_haystack.py \
    --env_conf test_needle/maskout_0.90.json \
    --chat_template llama-2 \
    --s_len 128 \
    --e_len 4096 \
    --step 128

python test_needle/viz.py \
    --env_conf test_needle/maskout_0.90.json \
