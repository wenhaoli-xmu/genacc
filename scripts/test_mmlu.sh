test_scripts=("maskout_0.98.json" "maskout_0.90.json")

for test_script in "${test_scripts[@]}"
do
    echo "Processing ${test_script}..."
    python mmlu/evaluate.py --env_conf "mmlu/${test_script}"
    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done