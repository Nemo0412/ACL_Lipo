#!/bin/bash
# 监控预测进度的脚本

echo "===================================================="
echo "药物效率分数预测进度监控"
echo "===================================================="
echo ""

# 检查进程是否在运行
if pgrep -f "predict_efficiency_local.py" > /dev/null; then
    echo "✓ 预测任务正在运行中..."
    echo ""
else
    echo "✗ 预测任务未运行"
    echo ""
fi

# 显示日志最后50行
echo "最新日志 (最后50行):"
echo "----------------------------------------------------"
tail -50 ../data/prediction_log.txt
echo "----------------------------------------------------"
echo ""

# 统计已处理的药物数
processed=$(grep -c "✓ 效率分数" ../data/prediction_log.txt 2>/dev/null || echo 0)
echo "已处理药物数: $processed / 2588"

if [ $processed -gt 0 ]; then
    percentage=$(echo "scale=2; $processed * 100 / 2588" | bc)
    echo "完成进度: ${percentage}%"
    
    # 估算剩余时间（假设平均0.5秒/药物）
    remaining=$((2588 - processed))
    estimated_seconds=$(echo "$remaining * 0.5" | bc)
    estimated_minutes=$(echo "scale=1; $estimated_seconds / 60" | bc)
    echo "预计剩余时间: 约 ${estimated_minutes} 分钟"
fi

echo ""
echo "===================================================="
echo "使用方法："
echo "  bash monitor_progress.sh     # 查看当前进度"
echo "  tail -f ../data/prediction_log.txt  # 实时查看日志"
echo "===================================================="
