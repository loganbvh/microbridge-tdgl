#!/bin/bash
#SBATCH --job-name=single_hole
#SBATCH --output=slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB
#SBATCH --time=24:00:00

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate tdgl

outdir=$GROUP_SCRATCH/microbridge-tdgl/data/$SLURM_JOB_ID
mkdir -p $outdir

pyscript=$HOME/microbridge-tdgl/scripts/single_hole.py

# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

python $pyscript \
    --output=$outdir/results.h5 \
    --ncpus=$SLURM_CPUS_PER_TASK \
    --width=500 \
    --height=500 \
    --radius=50 \
    --lambda_=200 \
    --xi=10 \
    --currents 550 650 30 \
    --fields 0 30 7 \
    --solve-time=700 \
    --eval-time=200

# Move the stdout log to the results directory
mv "slurm-${SLURM_JOB_ID}.out" $outdir
