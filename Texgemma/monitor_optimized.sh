#!/bin/bash
# 监控优化后的预测进度和效果

echo "=========================================="
echo "TXGemma 优化预测监控"
echo "=========================================="
echo ""

# 检查进程
PID=$(cat /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/prediction_pid.txt 2>/dev/null)
if ps -p $PID > /dev/null 2>&1; then
    echo "✓ 预测进程正在运行 (PID: $PID)"
    
    # CPU和内存使用
    ps -p $PID -o %cpu,%mem,etime,cmd | tail -1
else
    echo "✗ 预测进程未运行"
    exit 1
fi

echo ""
echo "=========================================="
echo "预测进度"
echo "=========================================="

# 统计完成数量
cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma
COMPLETED=$(grep -c '"ID"' result.json 2>/dev/null || echo "0")
TOTAL=10024
REMAINING=$((TOTAL - COMPLETED))
PROGRESS=$(echo "scale=2; $COMPLETED * 100 / $TOTAL" | bc)

echo "已完成: $COMPLETED / $TOTAL ($PROGRESS%)"
echo "剩余: $REMAINING"
echo ""

# 估算完成时间（基于最近的速度）
if [ -f "prediction_log_optimized.txt" ]; then
    START_TIME=$(grep "Start time:" prediction_log_optimized.txt | tail -1 | awk '{print $NF}')
    if [ ! -z "$START_TIME" ]; then
        # 计算从开始到现在的时间
        CURRENT_TIME=$(date +%s)
        START_TIMESTAMP=$(date -d "$START_TIME" +%s 2>/dev/null || echo $CURRENT_TIME)
        ELAPSED=$((CURRENT_TIME - START_TIMESTAMP))
        
        if [ $ELAPSED -gt 0 ] && [ $COMPLETED -gt 2250 ]; then
            NEW_COMPLETED=$((COMPLETED - 2250))
            SPEED=$(echo "scale=2; $NEW_COMPLETED / $ELAPSED" | bc)
            
            if [ $(echo "$SPEED > 0" | bc) -eq 1 ]; then
                REMAINING_SECONDS=$(echo "$REMAINING / $SPEED" | bc)
                REMAINING_HOURS=$(echo "scale=1; $REMAINING_SECONDS / 3600" | bc)
                
                echo "当前速度: $SPEED 分子/秒"
                echo "预计剩余时间: $REMAINING_HOURS 小时"
                echo ""
            fi
        fi
    fi
fi

echo "=========================================="
echo "优化效果统计"
echo "=========================================="

# 统计各分数的分布
echo "分数分布："
for score in {1..10}; do
    count=$(grep -c "\"efficiency_score\": $score" result.json 2>/dev/null || echo "0")
    if [ $count -gt 0 ]; then
        echo "  分数 $score: $count 个"
    fi
done

echo ""

# 统计有理由的分子数量
WITH_REASON=$(grep -c '"reason": "[^"]' result.json 2>/dev/null || echo "0")
WITHOUT_REASON=$(grep -c '"reason": ""' result.json 2>/dev/null || echo "0")

echo "理由生成统计："
echo "  有详细理由: $WITH_REASON ($((WITH_REASON * 100 / COMPLETED))%)"
echo "  无理由(快速): $WITHOUT_REASON ($((WITHOUT_REASON * 100 / COMPLETED))%)"
echo ""

# 计算实际加速比
if [ $WITHOUT_REASON -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; 1 / (1 - 0.75 * $WITHOUT_REASON / $COMPLETED)" | bc)
    echo "预计加速比: ${SPEEDUP}x (假设快速预测节省75%时间)"
fi

echo ""
echo "=========================================="
echo "最近的预测结果"
echo "=========================================="
tail -20 result.json | grep -E '"ID"|"efficiency_score"|"reason"' | tail -12

echo ""
echo "=========================================="
echo "监控命令："
echo "  tail -f prediction_log_optimized.txt"
echo "  watch -n 10 ./monitor_optimized.sh"
echo "=========================================="
