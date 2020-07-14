#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi
cd src &&
[ -f '../pretrained/balls_interaction.pth' ] && python main.py --task eval_balls --config configs/balls_interaction.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/balls_interaction.pth' val.mode 'test' val.eval_types '["generation"]' val.metrics '["med"]'
[ -f '../pretrained/balls_occlusion.pth' ] && python main.py --task eval_balls --config configs/balls_occlusion.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/balls_occlusion.pth' val.mode 'test' val.eval_types '["generation"]' val.metrics '["med"]'
[ -f '../pretrained/balls_two_layer.pth' ] && python main.py --task eval_balls --config configs/balls_two_layer.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/balls_two_layer.pth' val.mode 'test' val.eval_types '["generation"]' val.metrics '["med"]'
[ -f '../pretrained/balls_two_layer_dense.pth' ] && ython main.py --task eval_balls --config configs/balls_two_layer_dense.yaml resume True device 'cuda:0' resume_ckpt '../pretrained/balls_two_layer_dense.pth' val.mode 'test' val.eval_types '["generation"]' val.metrics '["med"]'
