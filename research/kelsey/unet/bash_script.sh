#!/bin/bash
#SBATCH --job-name=sudsaq
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task 10
#SBATCH --output=/auto/users/kelsen/slurm/%j.out
#SBATCH --time=11:00:00
#SBATCH --ntasks=1
#SBATCH --nodelist=oat4

# Load modules if needed

# Define source and destination directories
SRC_DIR="clpc278:/scratch-ssd/kelsen/sudsaq/data"
DEST_DIR="/scratch-ssd/kelsen/sudsaq/data"


# Print for logging
echo "Starting rsync at $(date)"

# Check if destination directory exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory does not exist. Creating and syncing files..."
    mkdir -p "$DEST_DIR"

    # Use rsync to copy files from SRC to DEST
    rsync -avh "$SRC_DIR/" "$DEST_DIR/"
    echo "rsync completed."
else
    echo "Destination directory already exists. Skipping rsync."
fi

source  /users/kelsen/miniconda3/etc/profile.d/conda.sh
conda activate aq

cd /auto/users/kelsen/code/sudsaq-kelsey-research/research/kelsey/unet

python run_pipeline.py --epochs 1 --optimizer adam --classes 1 --test_year 2019 --overfitting False --channels 51 --target bias --region NorthAmerica --model_type ensemble --data_dir /scratch-ssd/kelsen/sudsaq/data --save_dir /users/kelsen/sudsaq-results --val_percent 0.1 --analysis_month June --tag testing-changes-thesis --norm zscore --wandb_status online