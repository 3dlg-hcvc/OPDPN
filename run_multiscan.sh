#!/bin/bash
export PROJ_DIR=/home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/PC_motion_prediction
sbatch --exclude=cdr26,cdr27,cdr28,cdr29,cdr30,cdr31,cdr32,cdr33,cdr34,cdr35,cdr40,cdr104,cdr111,cdr905,cdr922,cdr199 $PROJ_DIR/train_script_multiscan.sh