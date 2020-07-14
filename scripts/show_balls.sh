#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi
cd src && \
python main.py --task vis_balls --config configs/balls_interaction.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/balls_interaction.pth' resultdir '../output/vis'
python main.py --task vis_balls --config configs/balls_occlusion.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/balls_occlusion.pth' resultdir '../output/vis'
python main.py --task vis_balls --config configs/balls_two_layer.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/balls_two_layer.pth' resultdir '../output/vis'
python main.py --task vis_balls --config configs/balls_two_layer_dense.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/balls_two_layer_dense.pth' resultdir '../output/vis'
