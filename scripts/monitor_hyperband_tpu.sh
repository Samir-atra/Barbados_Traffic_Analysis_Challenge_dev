#!/bin/bash
# Monitor the TPU Hyperband kernel on Kaggle

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow

echo "Monitoring Kaggle Kernel: barbados-traffic-ff-hyperband-tpu"
echo "============================================================"
echo ""

while true; do
    STATUS=$(kaggle kernels status samerattrah/barbados-traffic-ff-hyperband-tpu 2>&1)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$TIMESTAMP] $STATUS"
    
    # Check if kernel is complete or failed
    if echo "$STATUS" | grep -q "complete"; then
        echo ""
        echo "Kernel completed successfully!"
        echo "Downloading output..."
        kaggle kernels output samerattrah/barbados-traffic-ff-hyperband-tpu -p kaggle_logs/hyperband_tpu_output
        echo "Output saved to: kaggle_logs/hyperband_tpu_output"
        break
    elif echo "$STATUS" | grep -q "error\|failed"; then
        echo ""
        echo "Kernel failed. Check logs on Kaggle."
        break
    fi
    
    # Wait 30 seconds before checking again
    sleep 30
done
