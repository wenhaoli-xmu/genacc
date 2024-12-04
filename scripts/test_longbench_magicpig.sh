test_scripts=("llama2-7b-magicpig.json")

model_max_length=4096

for test_script in "${test_scripts[@]}"
do
    echo "Running prediction for ${test_script}..."
    python test_longbench/pred.py --env_conf "test_longbench/${test_script}" --model_max_length $model_max_length --magicpig

    echo "Evaluating model for ${test_script}..."
    python LongBench/eval.py --model "${test_script}"

    echo "Displaying results for ${test_script}..."
    cat "pred/${test_script}/result.json"

    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done