#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=brainnet-train
#SBATCH --output=brainnet_log.out
#SBATCH --error=brainnet_log.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


# Load modules if needed
# module load python/3.9 cuda/11.7

# Source conda
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate environment
conda activate wids-datathon

# Move into your project directory
cd /Users/Haley/Desktop/WiDs Datathon/WiDs Notebooks/

# NOW: Run the notebook non-interactively!
jupyter nbconvert --to notebook --execute GNN_MLP.ipynb --inplace --ExecutePreprocessor.timeout=600
