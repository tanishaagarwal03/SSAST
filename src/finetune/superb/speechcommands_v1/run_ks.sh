#!/bin/bash
#SBATCH --job-name="s3prl-ks"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=./log_ks_%j.txt

set -x

# --- 1. SETUP ENVIRONMENT ---
# MODIFY THIS: Point to your actual virtualenv
source ../../../venvssast/bin/activate
export TORCH_HOME=../../pretrained_models

# --- 2. UNZIP DATA TO SCRATCH ---
ZIP_PATH="../datasets/speech_commands_v0.01.zip"
SCRATCH_ROOT="/disk/scratch/${USER}/data"
mkdir -p ${SCRATCH_ROOT}

# Target directory for unzipped data
DATA_DIR="${SCRATCH_ROOT}/speech_commands_v0.01"

if [ ! -d "${DATA_DIR}" ]; then
    echo "Unzipping data to ${SCRATCH_ROOT}..."
    # If using .tar.gz, replace 'unzip -q' with 'tar -xzf' and adjust arguments
    unzip -q ${ZIP_PATH} -d ${SCRATCH_ROOT}
    # Handle case where zip doesn't have a parent folder
    # if the zip dumps files directly into SCRATCH_ROOT, you might need to move them to DATA_DIR
else
    echo "Data directory already exists at ${DATA_DIR}. Skipping unzip."
fi

# --- 3. RUN TRAINING ---
mdl=ssast_frame_base_1s 
lr=5e-5
expname=ks_${mdl}_${lr}
expdir=./exp/$expname
mkdir -p $expdir

python3 run_downstream.py -m train \
    --expdir ${expdir} \
    -n speech_commands \
    -u $mdl \
    -d speech_commands \
    -c config_ks.yaml \
    -s hidden_states \
    -o "config.optimizer.lr=$lr,config.downstream_expert.datarc.speech_commands_root='${DATA_DIR}'" \
    -f