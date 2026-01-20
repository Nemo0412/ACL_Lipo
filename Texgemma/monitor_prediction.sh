#!/bin/bash#!/bin/bash

# 监控 TXGemma-27B 预测进度

echo "=================================="

echo "TXGemma-27B-Chat Prediction Status"echo "=========================================="

echo "=================================="echo "TXGemma-27B Prediction Monitor"

echo ""echo "=========================================="

echo ""

# 检查进程

PID=$(ps aux | grep predict_txgemma_incremental.py | grep -v grep | awk '{print $2}')# 检查进程是否运行

if [ -z "$PID" ]; thenif [ -f "/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/prediction.pid" ]; then

    echo "❌ No process running"    PID=$(cat /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/prediction.pid)

    exit 1    if ps -p $PID > /dev/null 2>&1; then

fi        echo "✓ Prediction is RUNNING (PID: $PID)"

    else

echo "✅ Process running (PID: $PID)"        echo "✗ Prediction process not found (may have completed or crashed)"

ps aux | grep $PID | grep -v grep | awk '{printf "   CPU: %s%% | MEM: %.1fGB | Time: %s\n", $3, $6/1024/1024, $10}'    fi

echo ""else

    echo "✗ No PID file found. Prediction may not have been started."

# GPU状态fi

echo "🖥️  GPU Usage:"

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "   GPU %s: %s | %s\n", $1, $2, $3}'echo ""

echo ""echo "Latest Log Output:"

echo "----------------------------------------"

# 结果统计tail -30 /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/prediction_log.txt 2>/dev/null || echo "No log file yet"

if [ -f "/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json" ]; then

    COUNT=$(python3 -c "import json; print(len(json.load(open('/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json'))))" 2>/dev/null || echo "0")echo ""

    PERCENT=$(python3 -c "print(f'{$COUNT/10024*100:.2f}')" 2>/dev/null || echo "0")echo "----------------------------------------"

    echo "📊 Progress: $COUNT / 10024 ($PERCENT%)"echo ""

    

    if [ "$COUNT" -gt "0" ]; then# 检查输出文件

        echo ""if [ -f "/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json" ]; then

        echo "📈 Score Distribution (recent):"    LINES=$(wc -l < /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json)

        python3 << 'EOF' 2>/dev/null    echo "✓ result.json exists ($LINES lines)"

import jsonelse

try:    echo "✗ result.json not created yet"

    data = json.load(open('/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json'))fi

    recent = data[-100:] if len(data) > 100 else data

    scores = {}echo ""

    for item in recent:echo "GPU Status:"

        s = item['efficiency_score']echo "----------------------------------------"

        scores[s] = scores.get(s, 0) + 1nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

    for score in sorted(scores.keys(), reverse=True):

        bar = '█' * int(scores[score] / max(scores.values()) * 30)echo ""

        print(f"   Score {score:2d}: {scores[score]:3d} {bar}")
except:
    pass
EOF
        
        echo ""
        echo "🏆 Top 3 Current:"
        python3 << 'EOF' 2>/dev/null
import json
try:
    data = json.load(open('/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json'))
    sorted_data = sorted(data, key=lambda x: x['efficiency_score'], reverse=True)
    for i, item in enumerate(sorted_data[:3], 1):
        print(f"   #{i}: ID={item['ID']}, Score={item['efficiency_score']}")
        print(f"       {item['reason'][:70]}...")
except:
    pass
EOF
    fi
fi

echo ""
echo "=================================="
echo "Run: bash monitor_prediction.sh"
echo "=================================="
