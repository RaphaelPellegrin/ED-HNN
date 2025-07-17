#!/bin/bash
#SBATCH --job-name=EDGNN_training
#SBATCH --output=logs/edgnn_%j.out
#SBATCH --error=logs/edgnn_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rpellegrinext@fas.harvard.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths for cluster environment
DATA_DIR="/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/ED-HNN/data"
RAW_DATA_DIR="/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/ED-HNN/raw_data"
LOG_FILE="logs/training_log_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}.txt"
SLURM_OUTPUT="logs/slurm_${SLURM_JOB_ID}.out"
SLURM_ERROR="logs/slurm_${SLURM_JOB_ID}.err"

# Redirect all output to log files
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Create log file header
echo "=== EDGNN Training Log - $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "========================================"

# Load necessary modules (adjust based on your cluster setup)
echo "Loading modules..."
# Try different module names for Python
module load python 2>&1 || module load anaconda 2>&1 || echo "No Python module found, using system Python"
module load cuda/11.8 2>&1 || echo "CUDA module not found"

# Use specific conda environment for EDGNN
ENV_PREFIX="/n/home04/rpellegrinext/edhnn_env"

# Source conda
source /n/home04/rpellegrinext/miniconda3/etc/profile.d/conda.sh

# Check if environment exists and activate it
if [ -d "$ENV_PREFIX" ]; then
    echo "✅ Conda environment exists at $ENV_PREFIX"
    echo "Activating existing environment..."
    conda activate "$ENV_PREFIX"
    echo "✅ Successfully activated environment"
else
    echo "❌ Conda environment does not exist at $ENV_PREFIX"
    echo "Creating new environment..."
    conda create --prefix "$ENV_PREFIX" python=3.11 -y
    conda activate "$ENV_PREFIX"
    echo "✅ Created and activated new environment"
fi

# Check if PyTorch is already installed and working
echo "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyTorch not found or broken, reinstalling..."
    pip uninstall torch torch-scatter torch-sparse torch-cluster torch-geometric -y
    pip install torch
    pip install torch-scatter torch-sparse torch-cluster torch-geometric --index-url https://pytorch-geometric.com/whl/torch-2.7.1+cu121
fi

# Fix NumPy version first (downgrade to avoid compatibility issues)
echo "Fixing NumPy version..."
pip install "numpy<2"

# Reinstall PyTorch with a stable version
echo "Reinstalling PyTorch with stable version..."
pip uninstall torch torch-scatter torch-sparse torch-cluster torch-geometric -y
echo "PyTorch uninstallation completed"

echo "Installing PyTorch 2.0.1+cu118..."
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
echo "PyTorch installation completed"

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo "PyTorch verification completed"

# Install PyTorch Geometric packages with compatible versions
echo "Installing PyTorch Geometric packages..."
# Try CPU version first since CUDA wheels might not be available
pip install torch-scatter torch-sparse torch-cluster torch-geometric --index-url https://pytorch-geometric.com/whl/torch-2.0.1+cpu

# If that fails, try the default installation
if [ $? -ne 0 ]; then
    echo "CPU version failed, trying default installation..."
    pip install torch-scatter torch-sparse torch-cluster torch-geometric
fi
echo "PyTorch Geometric installation completed"

# Verify PyTorch Geometric installation
echo "Verifying PyTorch Geometric installation..."
python -c "import torch_geometric; print('PyTorch Geometric imported successfully')"
echo "PyTorch Geometric verification completed"

# Install other required packages if not already installed
echo "Checking other dependencies..."
python -c "import configargparse" 2>/dev/null || pip install configargparse

# Check CUDA availability
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available. GPU info:"
    nvidia-smi
    CUDA_ID=0
else
    echo "CUDA not available. Setting CUDA_ID to -1 (CPU mode)"
    CUDA_ID=-1
fi

# Check if data directories exist
echo "Checking data directories..."
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    exit 1
fi
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "ERROR: Raw data directory does not exist: $RAW_DATA_DIR"
    exit 1
fi
echo "Data directories found successfully."

# Check if cora dataset exists in raw_data
echo "Checking for cora dataset..."
if [ -d "$RAW_DATA_DIR/cocitation/cora" ]; then
    echo "Cora dataset found in cocitation directory."
elif [ -d "$RAW_DATA_DIR/coauthorship/cora" ]; then
    echo "Cora dataset found in coauthorship directory."
else
    echo "WARNING: Cora dataset not found in expected locations."
    echo "Available datasets in raw_data:"
    ls "$RAW_DATA_DIR"
fi

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo "SLURM output: $SLURM_OUTPUT"
echo "SLURM error: $SLURM_ERROR"

# Run the training command
echo "Running training command..."
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
    --raw_data_dir "$RAW_DATA_DIR"

# Check exit status
TRAINING_EXIT_CODE=$?
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE"
fi

# Final summary
echo "========================================"
echo "Job completed at $(date)"
echo "Training exit code: $TRAINING_EXIT_CODE"
echo "Log file: $LOG_FILE"
echo "SLURM output: $SLURM_OUTPUT"
echo "SLURM error: $SLURM_ERROR"
echo "========================================"

exit $TRAINING_EXIT_CODE 