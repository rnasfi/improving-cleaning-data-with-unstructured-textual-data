#!/usr/bin/env bash
# Set & move to home directory
source ./set_env.sh

script="./examples/allergens.py" # trials_design  trials_population allergens
# script="./tests/check_results.py --data_index=2"
if [ $# -eq 1 ] ; then
  script="$1"
fi

echo "Launching example script $script"
python3 $script