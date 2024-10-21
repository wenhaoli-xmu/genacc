test_scripts=("maskout_0.98.json" "maskout_0.95.json" "maskout_0.90.json" "maskout_0.80.json" "llama2-7b.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running test for ${test_script}..."
    python test_ppl/test.py --env_conf "test_ppl/${test_script}"
    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done