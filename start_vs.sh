#!/bin/bash
# 本地互踢：同时拉起两支队名不同的球队（各 7 人，9×14 场地）。
# 两队队名必须不同，否则服务器报错。用于 A/B 实测自家策略。
# 用法：./start_vs.sh [host] [port] [teamA] [teamB]
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-60000}
team_a=${3:-Freeloap}
team_b=${4:-Baseline}

echo "启动队伍 A：$team_a（7 人）"
for i in {1..7}; do
  python3 run_player.py --host $host --port $port -n $i -t $team_a -f 7v7 &
done

echo "启动队伍 B：$team_b（7 人）"
for i in {1..7}; do
  python3 run_player.py --host $host --port $port -n $i -t $team_b -f 7v7 &
done

echo "两队已启动。用 ./kill.sh 结束全部球员。"
