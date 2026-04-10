#!/bin/bash
#SBATCH --job-name=agibot_build
#SBATCH --output=agibot_build_%j.out
#SBATCH --error=agibot_build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=500G
#SBATCH --time=48:00:00

# AgiBotWorld-Beta Dataset Build Job
# This script submits the dataset building process to SLURM

echo "=================================================="
echo "AgiBotWorld-Beta Dataset Build Job"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Change to the script directory
cd /n/fs/robot-data/rlds_multithread

# Run the build script
bash build_agibot_large_dataset.sh

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "=================================================="
echo "Job Complete"
echo "=================================================="
echo "End time: $(date)"
echo "Exit status: $EXIT_STATUS"

# Print summary statistics
if [ -f "corrupted_episodes.json" ]; then
    echo ""
    echo "Corrupted episodes summary:"
    python3 -c "
import json
with open('corrupted_episodes.json', 'r') as f:
    data = json.load(f)
print(f'  Total corrupted episodes: {len(data[\"corrupted_episodes\"])}')
"
fi

exit $EXIT_STATUS
