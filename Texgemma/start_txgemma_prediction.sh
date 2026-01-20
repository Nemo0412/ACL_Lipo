#!/bin/bash
# 启动 TXGemma-27B-Predict 虚拟分子库预测

echo "=========================================="
echo "TXGemma-27B-Predict Virtual Library"
echo "mRNA Transfection Efficiency Prediction"
echo "=========================================="
echo ""
echo "Starting prediction at: $(date)"
echo ""

cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma

# 检查预处理数据是否存在
if [ ! -f "data/virtual_library_preprocessed.jsonl" ]; then
    echo "✗ Preprocessed data not found!"
    echo "Running data preprocessing first..."
    python3 preprocess_data.py
    echo ""
fi

# 运行预测
echo "Starting TXGemma prediction..."
nohup python3 predict_txgemma.py > prediction_log.txt 2>&1 &

PID=$!
echo "✓ Prediction started with PID: $PID"
echo "$PID" > prediction.pid

echo ""
echo "=========================================="
echo "Prediction Started"
echo "=========================================="
echo ""
echo "📁 Files:"
echo "   - Input: data/virtual_library_preprocessed.jsonl"
echo "   - Output: result.json (sorted by score)"
echo "   - Log: prediction_log.txt"
echo ""
echo "📊 Monitoring:"
echo "   tail -f prediction_log.txt"
echo "   bash monitor_prediction.sh"
echo ""
echo "⏱️  Estimated time: ~28 hours for 10,024 molecules"
echo "Expected completion: $(date -d '+28 hours' '+%Y-%m-%d %H:%M:%S')"
echo ""
