#!/bin/bash
# 创建独立的 RL 训练虚拟环境并安装依赖。
# 用法：bash rl_getup/setup.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv_rl"

echo "在 $VENV 创建虚拟环境..."
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip
echo "安装依赖（torch 体积较大，请耐心）..."
"$VENV/bin/python" -m pip install -r "$HERE/requirements.txt"

echo ""
echo "完成。后续命令用：$VENV/bin/python"
echo "  训练： $VENV/bin/python rl_getup/train.py"
echo "  导出： $VENV/bin/python rl_getup/export_onnx.py"
