#!/bin/bash
#Usage: ./run.sh GPU criterion
set -e
set -x 

export CUDA_VISIBLE_DEVICES=$1
criterion=$2
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOGS=logs
if [ ! -d $LOGS ]; then
    mkdir $LOGS
fi
LOG="${LOGS}/${criterion}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

if [ -n $criterion ]; then
    criterion="--criterion ${criterion}"
fi


python sl_training.py $criterion --iter 1
python sl_training.py $criterion --iter 1
for i in $(seq 2 3) 
do
    python greedy_search.py --iter $i --test
    python greedy_search.py --iter $i
    python sl_training.py $criterion --iter $i
    python joint_training.py $criterion --iter $i
done
