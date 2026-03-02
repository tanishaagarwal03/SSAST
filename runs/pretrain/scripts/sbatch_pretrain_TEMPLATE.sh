#!/bin/bash
#SBATCH --job-name=ssast-pretrain
#SBATCH --partition=PGR-Standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/s2211921/ssast/runs/pretrain/slurm_log/%x_%j.txt

set -euo pipefail
set -x

. /home/htang2/toolchain-20251006/toolchain.rc
source /home/s2211921/ssast/venvssast/bin/activate

# ---------- repo/run dirs ----------
REPO_ROOT=/home/s2211921/SSAST
RUN_ROOT=$REPO_ROOT/runs/pretrain
EXP_ROOT=$RUN_ROOT/exp
LOG_ROOT=$RUN_ROOT/slurm_log
mkdir -p "$EXP_ROOT" "$LOG_ROOT"

# ---------- scratch data paths ----------
SCRATCH=/disk/scratch/s2211921
LIBRI_PARENT=$SCRATCH/librispeech
LIBRI_ROOT=$LIBRI_PARENT/LibriSpeech
mkdir -p "$LIBRI_PARENT"

# ---------- tarballs in home ----------
TAR360=$HOME/libri-360.tar
TAR100=$HOME/train-clean-100.tar

# ---------- extract (only if missing) ----------
if [ ! -d "$LIBRI_ROOT/train-clean-360" ]; then
  echo "[INFO] train-clean-360 not found, extracting to $LIBRI_PARENT"
  tar -xf "$TAR360" -C "$LIBRI_PARENT"
else
  echo "[INFO] train-clean-360 exists, skipping extract"
fi

if [ ! -d "$LIBRI_ROOT/train-clean-100" ]; then
  echo "[INFO] train-clean-100 not found, extracting to $LIBRI_PARENT"
  tar -xf "$TAR100" -C "$LIBRI_PARENT"
else
  echo "[INFO] train-clean-100 exists, skipping extract"
fi

# ---------- JSON prep (only if missing) ----------
# NOTE: prep_librispeech.py writes to *fixed* locations under /disk/scratch/s2211921/librispeech
json_train="/disk/scratch/s2211921/librispeech/librispeech_tr360_cut.json"
json_val="/disk/scratch/s2211921/librispeech/librispeech_tr100_cut_test.json"

if [ ! -f "$json_train" ] || [ ! -f "$json_val" ]; then
  echo "[INFO] JSON not found, running prep_librispeech.py (fixed output paths)"
  python "$REPO_ROOT/src/prep_data/librispeech/prep_librispeech.py"
else
  echo "[INFO] JSON exists, skipping prep"
fi

# ---------- experiment params ----------
dataset=librispeech360
tr_data="$json_train"
te_data="$json_val"

dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
num_mel_bins=128

model_size=tiny
task=pretrain_mpmhb
mask_patch=400

# "frame-based" tokens (no time patching)
fshape=128
tshape=1
fstride=$fshape
tstride=$tshape

batch_size=64
lr=5e-4
lr_patience=2
epoch=5
freqm=0
timem=0
mixup=0
bal=none
num_workers=8

# MHB loss weights
num_clusters=512
mpmhb_weight=0.0
mpg_weight=10.0
mpc_weight=1.0
target_layer_idx=-1
cluster_update_freq=-1

# ---------- naming ----------
tag="${dataset}_${task}_${model_size}_f${fshape}t${tshape}_m${mask_patch}_e${epoch}_lr${lr}_mpmhb${mpmhb_weight}_mpg${mpg_weight}_mpc${mpc_weight}"
exp_dir="$EXP_ROOT/$tag"
mkdir -p "$exp_dir"

# ---------- run ----------
CUDA_CACHE_DISABLE=1 python -W ignore "$REPO_ROOT/src/run.py" \
  --dataset "$dataset" \
  --data-train "$tr_data" --data-val "$te_data" \
  --exp-dir "$exp_dir" \
  --label-csv "$REPO_ROOT/src/prep_data/fsd50k/class_labels_indices.csv" \
  --lr "$lr" --n-epochs "$epoch" --batch-size "$batch_size" --save_model True \
  --freqm "$freqm" --timem "$timem" --mixup "$mixup" --bal "$bal" \
  --tstride "$tstride" --fstride "$fstride" --fshape "$fshape" --tshape "$tshape" \
  --dataset_mean "$dataset_mean" --dataset_std "$dataset_std" \
  --target_length "$target_length" --num_mel_bins "$num_mel_bins" \
  --model_size "$model_size" --mask_patch "$mask_patch" --n-print-steps 100 \
  --task "$task" --lr_patience "$lr_patience" --epoch_iter 800 \
  --num_clusters "$num_clusters" --mpmhb_weight "$mpmhb_weight" --mpg_weight "$mpg_weight" --mpc_weight "$mpc_weight" \
  --target_layer_idx "$target_layer_idx" --cluster_update_freq "$cluster_update_freq" \
  --num-workers "$num_workers"