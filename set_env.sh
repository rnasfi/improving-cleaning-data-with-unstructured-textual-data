#!/usr/bin/env bash

export MLParkerCleanerHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd  )"
echo "MLParkerCleaner home directory: $MLParkerCleanerHOME"
export PYTHONPATH="$PYTHONPATH:$MLParkerCleanerHOME"
export PATH="$PATH:$MLParkerCleanerHOME"
echo $PATH
echo "Environment variables set!"
