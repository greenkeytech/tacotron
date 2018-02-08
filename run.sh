#!/bin/bash 
export OMP_NUM_THREADS=1
#export OMP_NUM_THREADS=$(grep -c proc /proc/cpuinfo)
if [ -z "$RESTART_FROM" ]; then 
  echo "" > /data/training/train.txt
  python proc_gkt_files.py 
  python train.py --base_dir="/data" --slack_url=$SLACK_URL 
else
  python train.py --base_dir="/data" --slack_url=$SLACK_URL --restore_step=$RESTART_FROM
fi
