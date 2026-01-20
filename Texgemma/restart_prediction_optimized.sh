#!/bin/bash
# 重启优化后的预测（仅对分数≥7或=1的分子生成详细理由）

cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma

echo "=========================================="
echo "重启优化后的预测任务"
echo "=========================================="
echo "优化策略："
echo "  - 所有分子先快速获取分数（50 tokens）"
echo "  - 只对分数≥7或=1的分子生成详细理由（512 tokens）"
echo "  - 分数2-6的分子不生成理由，节省时间"
echo "=========================================="
echo ""

# 检查已完成的数量
completed=$(grep -c '"ID"' result.json 2>/dev/null || echo "0")
echo "已完成分子数: $completed"
echo "总分子数: 10024"
echo "剩余: $((10024 - completed))"
echo ""

# 设置离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "启动预测..."
nohup python predict_txgemma_incremental.py > prediction_log_optimized.txt 2>&1 &

PID=$!
echo $PID > prediction_pid.txt
echo "预测进程已启动，PID: $PID"
echo ""
echo "监控命令："
echo "  tail -f prediction_log_optimized.txt"
echo "  ./monitor_txgemma.sh"
echo "  ps aux | grep predict_txgemma"
echo ""
echo "预计加速比: 3-4倍（假设70%的分子是中等分数）"
echo "预计总时间: 12-15小时（原来需要40小时）"
