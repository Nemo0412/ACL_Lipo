#!/bin/bash
# 监控TXGemma预测进度

echo "=================================="
echo "TXGemma 预测进度监控"
echo "=================================="
echo ""

# 检查进程是否在运行
if ps aux | grep -q "[p]redict_txgemma_incremental.py"; then
    echo "✓ 预测进程正在运行中"
    
    # 显示进程信息
    ps aux | grep "[p]redict_txgemma_incremental" | awk '{print "  PID: "$2" | CPU: "$3"% | MEM: "$4"% | TIME: "$10}'
    
    echo ""
    echo "最新日志:"
    echo "---"
    tail -20 prediction_log_incremental.txt | grep -v "^$"
    
    echo ""
    echo "---"
    
    # 统计已完成的分子数
    if [ -f "result.json" ]; then
        completed=$(grep -c '"ID"' result.json 2>/dev/null || echo "0")
        total=10024
        percentage=$(echo "scale=1; $completed * 100 / $total" | bc)
        remaining=$((total - completed))
        
        echo ""
        echo "预测进度:"
        echo "  已完成: $completed / $total 分子 ($percentage%)"
        echo "  剩余: $remaining 分子"
        
        # 估算剩余时间（假设每个分子15秒）
        if [ "$completed" -gt 0 ]; then
            estimated_seconds=$((remaining * 15))
            estimated_hours=$((estimated_seconds / 3600))
            estimated_mins=$(((estimated_seconds % 3600) / 60))
            echo "  预计剩余时间: ~${estimated_hours}小时${estimated_mins}分钟"
        fi
    fi
    
else
    echo "✗ 预测进程未运行"
    
    # 检查是否已完成
    if [ -f "result.json" ]; then
        completed=$(grep -c '"ID"' result.json 2>/dev/null || echo "0")
        if [ "$completed" -ge 10024 ]; then
            echo ""
            echo "✓ 预测已完成！"
            echo "   结果文件: result.json"
            echo "   总计: $completed 分子"
        else
            echo ""
            echo "预测中断，已完成 $completed / 10024 分子"
            echo ""
            echo "重新启动命令:"
            echo "  cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma"
            echo "  nohup python3 predict_txgemma_incremental.py > prediction_log_incremental.txt 2>&1 &"
        fi
    fi
fi

echo ""
echo "=================================="
echo "命令:"
echo "  实时查看日志: tail -f prediction_log_incremental.txt"
echo "  查看结果: cat result.json | jq '.[] | {ID, efficiency_score}' | head -20"
echo "=================================="
