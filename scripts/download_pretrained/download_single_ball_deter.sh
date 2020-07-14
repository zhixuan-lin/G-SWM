#!/usr/bin/env bash
if [ ! -d scripts ]; then
  echo 'please run this script from project root'
  exit 0
fi
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16FiJU1loLFRxzvyXxZCtfIaeAOwia4au' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16FiJU1loLFRxzvyXxZCtfIaeAOwia4au" -O pretrained/single_ball_deter.pth && rm -rf /tmp/cookies.txt

