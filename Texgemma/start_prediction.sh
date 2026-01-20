#!/bin/bash
# 启动 TXGemma-27B 虚拟分子库预测

echo "=========================================="
echo "TXGemma-27B Virtual Library Prediction"
echo "=========================================="
echo ""
echo "Starting prediction at: $(date)"
echo ""

cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma

# 运行预测
nohup python3 predict_virtual_library.py > prediction_log.txt 2>&1 &

PID=$!
echo "Prediction started with PID: $PID"
echo "$PID" > prediction.pid

echo ""
echo "To monitor progress:"
echo "  tail -f /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/prediction_log.txt"
echo ""
echo "To check status:"
echo "  bash monitor_prediction.sh"
echo ""
echo "Estimated completion time: ~27-28 hours for 10000 molecules"
echo ""
