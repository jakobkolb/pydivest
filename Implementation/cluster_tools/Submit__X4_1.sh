#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=X4_scan
#SBATCH --output=X4_scan_%j.out
#SBATCH --error=X4_scan_%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=16

module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

echo "_______________________________________________"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "_______________________________________________"

cd ../
srun -n $SLURM_NTASKS python X4_phase_transition.py 1
