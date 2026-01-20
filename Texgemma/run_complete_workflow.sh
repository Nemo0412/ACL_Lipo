#!/bin/bash
# 完整工作流：数据预处理 → 模型预测

echo "=========================================="
echo "TXGemma Virtual Library Prediction"
echo "Complete Workflow"
echo "=========================================="
echo ""

cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma

# Step 1: 数据预处理
echo "Step 1: Data Preprocessing"
echo "------------------------------------------"

if [ ! -f "data/virtual_library_preprocessed.jsonl" ]; then
    echo "Running data preprocessing..."
    python3 preprocess_data.py
    
    if [ $? -ne 0 ]; then
        echo "✗ Preprocessing failed!"
        exit 1
    fi
    echo "✓ Preprocessing completed"
else
    echo "✓ Preprocessed data already exists"
fi

echo ""

# Step 2: 模型预测
echo "Step 2: Model Prediction"
echo "------------------------------------------"
echo "Starting prediction at: $(date)"
echo ""

nohup python3 predict_virtual_library_v2.py > prediction_log.txt 2>&1 &

PID=$!
echo "✓ Prediction started with PID: $PID"
echo "$PID" > prediction.pid

echo ""
echo "=========================================="
echo "Workflow Started Successfully"
echo "=========================================="
echo ""
echo "📁 Files:"
echo "   - Preprocessed data: data/virtual_library_preprocessed.jsonl"
echo "   - Log file: prediction_log.txt"
echo "   - PID file: prediction.pid"
echo ""
echo "📊 Monitoring:"
echo "   View real-time log:  tail -f prediction_log.txt"
echo "   Check status:        bash monitor_prediction.sh"
echo "   View GPU usage:      watch -n 1 nvidia-smi"
echo ""
echo "⏱️  Estimated completion:"
echo "   ~27-30 hours for 10,000 molecules"
echo "   Check back at: $(date -d '+28 hours' '+%Y-%m-%d %H:%M:%S')"
echo ""
