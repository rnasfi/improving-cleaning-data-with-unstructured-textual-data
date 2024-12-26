#!/usr/bin/env bash
# Set & move to home directory
source ./set_env.sh

script="./example/allergens.py" # trials_design  trials_population allergens
if [ $# -eq 1 ] ; then
  script="$1"
fi

echo "Launching example script $script"
python3 $script