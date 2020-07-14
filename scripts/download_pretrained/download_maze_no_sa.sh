#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17UsxnpC_f9ZvnDro-mCR6p8100PpXXJZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17UsxnpC_f9ZvnDro-mCR6p8100PpXXJZ" -O pretrained/maze_no_sa.pth && rm -rf /tmp/cookies.txt

