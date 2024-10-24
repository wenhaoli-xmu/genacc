test_scripts=("maskout90.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running lmeval for ${test_script}..."
    python test_lmeval/lmeval.py --env_conf "test_lmeval/${test_script}"
    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done