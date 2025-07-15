#!/bin/bash

# EDGNN Training Script
# This script runs the EDGNN model training on the cora dataset

# Configuration
CUDA_ID=0  # Change this to your desired CUDA device ID
DATA_DIR="/Users/pellegrinraphael/Desktop/Repos_GNN/ED-HNN/data"
RAW_DATA_DIR="/Users/pellegrinraphael/Desktop/Repos_GNN/ED-HNN/raw_data"
LOG_FILE="training_log_$(date +%Y%m%d_%H%M%S).txt"

# Create log file
echo "=== EDGNN Training Log - $(date) ===" > "$LOG_FILE"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available. Using CUDA device: $CUDA_ID" | tee -a "$LOG_FILE"
else
    echo "CUDA not available. Setting CUDA_ID to -1 (CPU mode)" | tee -a "$LOG_FILE"
    CUDA_ID=-1
fi

# Check if data directories exist
echo "Checking data directories..." | tee -a "$LOG_FILE"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR" | tee -a "$LOG_FILE"
    exit 1
fi
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "ERROR: Raw data directory does not exist: $RAW_DATA_DIR" | tee -a "$LOG_FILE"
    exit 1
fi
echo "Data directories found successfully." | tee -a "$LOG_FILE"

# Check if cora dataset exists in raw_data
echo "Checking for cora dataset..." | tee -a "$LOG_FILE"
if [ -d "$RAW_DATA_DIR/cocitation/cora" ]; then
    echo "Cora dataset found in cocitation directory." | tee -a "$LOG_FILE"
elif [ -d "$RAW_DATA_DIR/coauthorship/cora" ]; then
    echo "Cora dataset found in coauthorship directory." | tee -a "$LOG_FILE"
else
    echo "WARNING: Cora dataset not found in expected locations." | tee -a "$LOG_FILE"
    echo "Available datasets in raw_data:" | tee -a "$LOG_FILE"
    ls "$RAW_DATA_DIR" | tee -a "$LOG_FILE"
fi

echo "Starting training..." | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training command and log all output
python train.py \
    --method EDGNN \
    --dname cora \
    --All_num_layers 1 \
    --MLP_num_layers 0 \
    --MLP2_num_layers 0 \
    --MLP3_num_layers 1 \
    --Classifier_num_layers 1 \
    --MLP_hidden 256 \
    --Classifier_hidden 256 \
    --aggregate mean \
    --restart_alpha 0.0 \
    --lr 0.001 \
    --wd 0 \
    --epochs 500 \
    --runs 10 \
    --cuda $CUDA_ID \
    --data_dir "$DATA_DIR" \
    --raw_data_dir "$RAW_DATA_DIR" 2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!" | tee -a "$LOG_FILE"
else
    echo "Training failed with exit code $?" | tee -a "$LOG_FILE"
fi

echo "Log file saved to: $LOG_FILE" 