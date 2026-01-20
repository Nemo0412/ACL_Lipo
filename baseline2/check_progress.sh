#!/bin/bash
# 检查 TXGemma-9B 评估进度

echo "=== TXGemma-9B 评估进度 ==="
echo ""

# 检查进程状态
if ps aux | grep -v grep | grep "evaluate_gemma_9b.py" > /dev/null; then
    echo "✓ 评估进程正在运行"
else
    echo "✗ 评估进程未运行"
fi

echo ""
echo "=== 最近的评估结果 (最后20行) ==="
tail -20 /mnt/3fs/dots-pretrain/leshu/workspace/baseline2/gemma_evaluation_full.txt 2>/dev/null || echo "日志文件尚未生成"

echo ""
echo "=== 评估统计 ==="
if [ -f /mnt/3fs/dots-pretrain/leshu/workspace/baseline2/gemma_evaluation_full.txt ]; then
    eff_count=$(grep -c "^\[.*\] .*Pred:" /mnt/3fs/dots-pretrain/leshu/workspace/baseline2/gemma_evaluation_full.txt 2>/dev/null || echo "0")
    echo "已评估样本数: $eff_count"
    
    if grep -q "### TOXICITY EVALUATION ###" /mnt/3fs/dots-pretrain/leshu/workspace/baseline2/gemma_evaluation_full.txt 2>/dev/null; then
        echo "状态: Toxicity 评估中"
    elif [ "$eff_count" -ge 600 ]; then
        echo "状态: Efficiency 评估完成"
    else
        echo "状态: Efficiency 评估中 ($eff_count/600)"
    fi
fi

echo ""
echo "提示: 运行 'bash /mnt/3fs/dots-pretrain/leshu/workspace/baseline2/check_progress.sh' 查看最新进度"
