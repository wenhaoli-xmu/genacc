test_scripts=("maskout_0.98.json" "maskout_0.90.json" "llama2-7b.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running prediction for ${test_script}..."
    python test_longbench/pred.py --env_conf "test_longbench/${test_script}"

    echo "Evaluating model for ${test_script}..."
    python LongBench/eval.py --model "${test_script}"

    echo "Displaying results for ${test_script}..."
    cat "pred/${test_script}/result.json"

    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done