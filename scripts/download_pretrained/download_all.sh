#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi

sh scripts/download_pretrained/download_balls_interaction.sh
sh scripts/download_pretrained/download_balls_occlusion.sh
sh scripts/download_pretrained/download_balls_two_layer.sh
sh scripts/download_pretrained/download_balls_two_layer_dense.sh
sh scripts/download_pretrained/download_maze.sh
sh scripts/download_pretrained/download_maze_no_aoe.sh
sh scripts/download_pretrained/download_maze_no_mu.sh
sh scripts/download_pretrained/download_maze_no_sa.sh
sh scripts/download_pretrained/download_obj3d.sh
sh scripts/download_pretrained/download_single_ball.sh
sh scripts/download_pretrained/download_single_ball_deter.sh
