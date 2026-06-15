#!/bin/bash
# 启动完整球队：7 名球员，比赛场地 9×14（hl_adult）。
# 用法：./start.sh [host] [port] [team]
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-60000}
team=${3:-Freeloap}

for i in {1..7}; do
  python3 run_player.py --host $host --port $port -n $i -t $team -f hl_adult &
done
