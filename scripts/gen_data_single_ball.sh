#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi
python scripts/dataset_single_ball/gen.py --split 10000 100 100 --color 1 0 0
