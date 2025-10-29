#!/bin/bash
# ====================================================
# Filename: monitor_system.sh
# 功能: 同时监控 GPU、CPU、内存 和 IO 状况
# 作者: ChatGPT 调优模板（校正 pidstat 参数，15s刷新）
# ====================================================

INTERVAL=15   # 每隔 15 秒刷新

# 检查依赖
command -v nvidia-smi >/dev/null 2>&1 || { echo "错误: 未安装 nvidia-smi"; exit 1; }
command -v pidstat >/dev/null 2>&1 || { echo "错误: 未安装 sysstat (pidstat)"; exit 1; }
command -v iostat >/dev/null 2>&1 || { echo "错误: 未安装 iostat"; exit 1; }

# 查找当前用户的所有 python 进程
PY_PIDS=$(pgrep -u "$USER" python | tr '\n' ',' | sed 's/,$//')
if [ -z "$PY_PIDS" ]; then
    echo "未检测到 python 训练进程，将监控系统整体 CPU 使用率"
fi

clear
echo "====================================================="
echo " 系统资源动态监控 (GPU + CPU + IO)"
echo " 每 ${INTERVAL}s 刷新一次 — Ctrl+C 退出"
echo "====================================================="

while true; do
    echo ""
    echo "###################### 时间: $(date '+%Y-%m-%d %H:%M:%S') ######################"
    
    echo ""
    echo "【GPU 利用情况】"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits

    echo ""
    echo "【CPU 利用情况】"
    if [ -n "$PY_PIDS" ]; then
        echo "(监控 PID: $PY_PIDS)"
        # 修正 pidstat 参数调用顺序和引号
        pidstat -p "$PY_PIDS" $INTERVAL 1 | awk 'NR==1 || NR>=4 {print}'
    else
        mpstat -P ALL 1 1 | tail -n +4
    fi

    echo ""
    echo "【I/O 利用情况】"
    iostat -x 1 1 | sed -n '1,10p'

    echo ""
    echo "【内存使用情况】"
    free -h

    echo -e "\n=========================================================================="
    sleep $INTERVAL
done