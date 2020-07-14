#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi

cd src
[ -f '../pretrained/maze.pth' ] && python main.py --task eval_maze --config configs/maze.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/maze.pth' val.mode 'test'
[ -f '../pretrained/maze_no_aoe.pth' ] && python main.py --task eval_maze --config configs/ablation/maze_no_aoe.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/maze_no_aoe.pth' val.mode 'test'
[ -f '../pretrained/maze_no_mu.pth' ] && python main.py --task eval_maze --config configs/ablation/maze_no_mu.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/maze_no_mu.pth' val.mode 'test'
[ -f '../pretrained/maze_no_sa.pth' ] && python main.py --task eval_maze --config configs/ablation/maze_no_sa.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/maze_no_sa.pth' val.mode 'test'

cd ..
python scripts/plot_maze_ablation.py
