#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi
cd src && \
python main.py --task vis_3d --config configs/obj3d.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/obj3d.pth' resultdir '../output/vis'
