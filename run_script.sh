#!/usr/bin/env bash

set -x
python main.py --env-name "Hopper-v2" --config-file "default_config.yaml" --experts-dir <add-path-here> --seed=$RANDOM
