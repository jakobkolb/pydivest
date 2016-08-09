#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --qos=short
#SBATCH --job-name=X3_kd
#SBATCH --output=X3_kd_%j.out
#SBATCH --error=X3_kd_%j.err
#SBATCH --time=0-12
#SBATCH --nodes=2
#SBATCH --tasks-per-node=16


module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

echo "_______________________________________________"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "_______________________________________________"

cd ../
srun -n $SLURM_NTASKS python X3_kd.py
