#!/usr/bin/env bash
# Set & move to home directory
source ./set_env.sh

# script="./examples/trials_design.py" 
# script="./examples/trials_population.py" 
script="./examples/allergens.py" 
# script="./tests/check_results.py --data_index=2"
if [ $# -eq 1 ] ; then
  script="$1"
fi

echo "Launching example script $script"
python3 $script