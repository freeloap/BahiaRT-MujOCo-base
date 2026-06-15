#!/bin/bash
# 点球阶段上场：仅 1 名守门员（#1）+ 1 名射手（#7）。
# 规则 2.5(9)/2.7 要求每队提供 start_penalty.sh。
# 用法：./start_penalty.sh [host] [port] [team]
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-60000}
team=${3:-Freeloap}

# 守门员
python3 run_player.py --host $host --port $port -n 1 -t $team -f 7v7 &
# 射手
python3 run_player.py --host $host --port $port -n 7 -t $team -f 7v7 &
