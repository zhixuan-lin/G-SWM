#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi

python scripts/dataset_balls/gen.py --output-dir './data'
