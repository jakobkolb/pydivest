#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=X1_divest_trial
#SBATCH --output=X1__trial_%j.out
#SBATCH --error=X1__trial_%j.err
#SBATCH --nodes=2
#SBATCH --tasks-per-node=16

module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

echo "_______________________________________________"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "_______________________________________________"

srun -n $SLURM_NTASKS python X1_test_experiment.py 0
