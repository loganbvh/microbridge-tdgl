#!/bin/bash
#SBATCH --job-name=single_hole_array
#SBATCH --array=0-100
#SBATCH --output=slurm-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=24:00:00

set -x

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate tdgl

outdir=$GROUP_SCRATCH/microbridge-tdgl/data/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir

pyscript=$HOME/microbridge-tdgl/scripts/single_hole_array.py

# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

python $pyscript \
    --output=$outdir/results-${SLURM_ARRAY_TASK_ID}.h5 \
    --field=40 \
    --currents 150 250 $SLURM_ARRAY_TASK_COUNT \
    --index=$SLURM_ARRAY_TASK_ID \
    --width=1000 \
    --height=500 \
    --radius=50 \
    --lambda_=450 \
    --xi=4.5 \
    --max-edge-length=3 \
    --solve-time=2000 \
    --eval-time=250

# Move the stdout log to the results directory
mkdir $outdir/logs
mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs
