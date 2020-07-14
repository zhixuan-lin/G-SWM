#!/usr/bin/env bash
cd $(dirname "$0")
FILELIST=$(find . -type f -name '*.sh' | grep -vE 'use_(cpu|gpu)')
for FILE in $FILELIST; do
  # Linux: use -i''
  sed -i '' -E 's/cpu/cuda:0/g' $FILE
done
