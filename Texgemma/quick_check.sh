#!/bin/bash
README.md __pycache__ check_progress.sh check_raw_output.py data download_log.txt model_comparison_analysis.md monitor_download.sh monitor_optimized.sh monitor_prediction.sh monitor_txgemma.sh predict_txgemma.py predict_txgemma_incremental.py predict_virtual_library.py predict_virtual_library_v2.py prediction.pid prediction_log.txt prediction_log_incremental.txt prediction_log_new.txt prediction_log_optimized.txt prediction_pid.txt preprocess_data.py restart_prediction_optimized.sh result.json result_old_predict_model.json result_summary.txt run_complete_workflow.sh start_prediction.sh start_txgemma_prediction.sh test_model_simple.py test_output.txt README.md __pycache__ check_progress.sh check_raw_output.py data download_log.txt model_comparison_analysis.md monitor_download.sh monitor_optimized.sh monitor_prediction.sh monitor_txgemma.sh predict_txgemma.py predict_txgemma_incremental.py predict_virtual_library.py predict_virtual_library_v2.py prediction.pid prediction_log.txt prediction_log_incremental.txt prediction_log_new.txt prediction_log_optimized.txt prediction_pid.txt preprocess_data.py restart_prediction_optimized.sh result.json result_old_predict_model.json result_summary.txt run_complete_workflow.sh start_prediction.sh start_txgemma_prediction.sh test_model_simple.py test_output.

cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma

COMPLETED=$(grep -c '"ID"' result.json 2>/dev/null || echo "0")
TOTAL=10024

echo "=========================================="
echo "已完成: $COMPLETED / $TOTAL ($(awk "BEGIN {printf \"%.2f\", $COMPLETED*100/$TOTAL}")%)"
echo "=========================================="
echo ""
echo "最后完成的3个分子:"
tail -60 result.json | grep '"ID"' | tail -3 | while read line; do
    ID=$(echo $line | grep -oP '(?<="ID": )\d+')
    echo "  ID: $ID"
done
echo ""
echo "当前正在处理: ID $(($ID + 1)) 左右"
