#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi
cd src && \
python main.py --task vis_maze --config configs/maze.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/maze.pth' resultdir '../output/vis'
