#!/bin/bash
#SBATCH --job-name=E5_tiny_patch_mpmhb
#SBATCH --partition=Teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/s2211921/SSAST/runs/pretrain/slurm_log/%x_%j.txt

set -x

. /home/htang2/toolchain-20251006/toolchain.rc
source /home/s2211921/SSAST/venvssast/bin/activate

REPO=/home/s2211921/SSAST
EXP_ROOT=$REPO/runs/pretrain/exp
mkdir -p "$EXP_ROOT" "$REPO/runs/pretrain/slurm_log"

SCRATCH=/disk/scratch/s2211921
LIBRI_PARENT=$SCRATCH/librispeech
LIBRI_ROOT=$LIBRI_PARENT/LibriSpeech
mkdir -p "$LIBRI_PARENT"

TAR360=$HOME/libri-360.tar
TAR100=$HOME/train-clean-100.tar

if [ ! -d "$LIBRI_ROOT/train-clean-360" ]; then
  tar -xf "$TAR360" -C "$LIBRI_PARENT"
fi
if [ ! -d "$LIBRI_ROOT/train-clean-100" ]; then
  tar -xf "$TAR100" -C "$LIBRI_PARENT"
fi

json_train="$SCRATCH/librispeech/librispeech_tr360_cut.json"
json_val="$SCRATCH/librispeech/librispeech_tr100_cut_test.json"
if [ ! -f "$json_train" ] || [ ! -f "$json_val" ]; then
  python "$REPO/src/prep_data/librispeech/prep_librispeech.py"
fi

dataset=librispeech360
dataset_mean=-4.2677393
dataset_std=4.5689974
target_length=1024
num_mel_bins=128

model_size=tiny
task=pretrain_mpmhb

fshape=128
tshape=2
fstride=$fshape
tstride=$tshape

mask_patch=400

batch_size=64
lr=5e-4
epoch=5
lr_patience=2
freqm=0
timem=0
mixup=0
bal=none
num_workers=8

num_clusters=512
mpmhb_weight=1.0
mpc_weight=0.0
mpg_weight=0.0
target_layer_idx=-1
cluster_update_freq=-1

TAG="E5_${dataset}_${task}_${model_size}_F${fshape}_T${tshape}_mask${mask_patch}_K${num_clusters}_mhb${mpmhb_weight}_mpc${mpc_weight}_mpg${mpg_weight}_layer${target_layer_idx}"
exp_dir="$EXP_ROOT/$TAG"
mkdir -p "$exp_dir"

CUDA_CACHE_DISABLE=1 python -W ignore "$REPO/src/run.py" \
  --dataset "$dataset" \
  --data-train "$json_train" --data-val "$json_val" \
  --exp-dir "$exp_dir" \
  --label-csv "$REPO/src/pretrain/data/class_labels_indices.csv" \
  --lr "$lr" --n-epochs "$epoch" --batch-size "$batch_size" --save_model True \
  --freqm "$freqm" --timem "$timem" --mixup "$mixup" --bal "$bal" \
  --fshape "$fshape" --tshape "$tshape" --fstride "$fstride" --tstride "$tstride" \
  --dataset_mean "$dataset_mean" --dataset_std "$dataset_std" \
  --target_length "$target_length" --num_mel_bins "$num_mel_bins" \
  --model_size "$model_size" --mask_patch "$mask_patch" --n-print-steps 100 \
  --task "$task" --lr_patience "$lr_patience" --epoch_iter 800 \
  --num_clusters "$num_clusters" \
  --mpmhb_weight "$mpmhb_weight" --mpc_weight "$mpc_weight" --mpg_weight "$mpg_weight" \
  --target_layer_idx "$target_layer_idx" --cluster_update_freq "$cluster_update_freq" \
  --noise False \
  --num-workers "$num_workers"