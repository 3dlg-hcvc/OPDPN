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
python $PROJ_DIR/train.py --config-file $PROJ_DIR/configs/bmcc.yaml --gtbbx --gtcat --output-dir /home/hanxiao/scratch/proj-motionnet/pc_output --data-path $DATASET_DIR/MotionDataset_h5_6.11 --input-format depth --model_attr_path $PROJ_DIR/scripts/data/data_statistics/urdf-attr.json --flip_prob 0.5 --motion_weights 1 8 8 --opts MODEL.WEIGHTS /scratch/hanxiao/proj-motionnet/train_output/done/motion_finetuning_cc_0.001_60000_0.5/finetune_cc_depth_1___1_8_8/model_final.pth SOLVER.BASE_LR 0.0005 SOLVER.MAX_ITER 60000 SOLVER.STEPS '(36000, 48000)' SOLVER.CHECKPOINT_PERIOD 5000 INPUT.RNG_SEED 2 