#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:p100:1         # Number of GPUs (per node)
#SBATCH --mem=32000               # memory (per node)
#SBATCH --time=2-23:00            # time (DD-HH:MM)
#SBATCH --cpus-per-task=6         # Number of CPUs (per task)
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shawn_jiang@sfu.ca
#SBATCH --output=/home/hanxiao/scratch/proj-motionnet/pc_output/%x_%j.out
#SBATCH --job-name=pc_baseline
echo 'Start'

echo 'ENV Start'

module load StdEnv/2020  intel/2020.1.217
module load python/3.7
module load cuda/11.0
module load cudnn/8.0.3

source /home/hanxiao/scratch/proj-motionnet/pc_env/bin/activate

echo 'Job Start'
python train.py --train_path /home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/Dataset/dataset/train.h5 --test_path /home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/Dataset/dataset/val.h5 --output_dir /home/hanxiao/scratch/proj-motionnet/pc_output