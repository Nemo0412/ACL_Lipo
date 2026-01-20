#!/bin/bash

echo "==================================="
echo "TXGemma-27B Prediction Progress"
echo "==================================="
echo ""

# 检查进程状态
echo "📊 Process Status:"
ps aux | grep predict_txgemma.py | grep -v grep | awk '{printf "  PID: %s | CPU: %s%% | MEM: %.1fGB | Time: %s\n", $2, $3, $6/1024/1024, $10}'
echo ""

# 检查GPU使用
echo "🖥️  GPU Usage:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "  GPU %s: %s / %s | Utilization: %s\n", $1, $2, $3, $4}'
echo ""

# 检查是否有结果文件
if [ -f "/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json" ]; then
    echo "📁 Result File:"
    COUNT=$(cat /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json | grep -o '"ID"' | wc -l)
    echo "  ✓ Found result.json with $COUNT predictions"
    echo "  Progress: $COUNT / 10024 ($(echo "scale=2; $COUNT*100/10024" | bc)%)"
    echo ""
fi

# 显示最新日志
echo "📝 Latest Log (last 10 lines):"
tail -10 /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/prediction_log.txt | sed 's/^/  /'
echo ""

echo "==================================="
echo "Use: bash check_progress.sh"
echo "==================================="
