#!/bin/bash
# 监控模型下载进度

echo "=================================="
echo "TXGemma-27B 模型下载监控"
echo "=================================="
echo ""

# 检查下载进程
if ps aux | grep -q "[h]uggingface-cli.*txgemma-27b"; then
    echo "✓ 下载进程正在运行中"
    ps aux | grep "[h]uggingface-cli.*txgemma-27b" | awk '{print "  PID: "$2" | CPU: "$3"% | MEM: "$4"%"}'
    echo ""
fi

# 显示下载日志的最新内容
if [ -f "download_log.txt" ]; then
    echo "最新下载进度:"
    echo "---"
    tail -20 download_log.txt | grep -E "Fetching|Download|Moving|safetensors" | tail -10
    echo "---"
    echo ""
fi

# 检查已下载的模型文件
model_dir="$HOME/.cache/huggingface/hub/models--google--txgemma-27b-chat"
if [ -d "$model_dir" ]; then
    echo "已下载文件统计:"
    
    # 统计safetensors文件
    safetensors_count=$(find "$model_dir" -name "*.safetensors" -type f 2>/dev/null | wc -l)
    echo "  Safetensors文件: $safetensors_count / 12"
    
    # 总大小
    total_size=$(du -sh "$model_dir" 2>/dev/null | awk '{print $1}')
    echo "  已下载大小: $total_size"
    
    # 预计总大小约54GB
    echo "  预计总大小: ~54GB"
    echo ""
fi

# 检查是否下载完成
if [ -d "$model_dir/snapshots" ] && [ $(find "$model_dir" -name "*.safetensors" -type f 2>/dev/null | wc -l) -eq 12 ]; then
    echo "✓ 模型下载完成！"
    echo ""
    echo "可以开始运行预测了:"
    echo "  cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma"
    echo "  nohup python3 predict_txgemma_incremental.py > prediction_log_incremental.txt 2>&1 &"
else
    echo "下载进行中，请耐心等待..."
    
    # 估算剩余时间
    if [ -d "$model_dir" ]; then
        current_size_bytes=$(du -sb "$model_dir" 2>/dev/null | awk '{print $1}')
        target_size_bytes=$((54 * 1024 * 1024 * 1024))  # 54GB
        
        if [ "$current_size_bytes" -gt 0 ]; then
            progress=$((current_size_bytes * 100 / target_size_bytes))
            echo "  进度: ~${progress}%"
        fi
    fi
fi

echo ""
echo "=================================="
echo "实时查看下载日志: tail -f download_log.txt"
echo "=================================="
