#!/usr/bin/env bash
cd $(dirname "$0")
FILELIST=$(find . -type f -name '*.sh' | grep -vE 'use_(cpu|gpu)')
for FILE in $FILELIST; do
  # Linux: use -i''
  sed -i '' -E 's/cuda:0/cpu/g' $FILE
done
