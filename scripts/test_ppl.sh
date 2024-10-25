test_scripts=(
    "llama2-7b-chat.json" 
    "llama2-7b-chat-maskout98.json" 
    "llama2-7b-chat-maskout95.json"
    "llama2-7b-chat-maskout90.json"
    "llama2-7b-chat-maskout80.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running test for ${test_script}..."
    python test_ppl/test.py --env_conf "test_ppl/${test_script}"
    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done