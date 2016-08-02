#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=X3_psTest
#SBATCH --output=X3_psTest_%j.out
#SBATCH --error=X3_psTest_%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=16

module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

echo "_______________________________________________"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "_______________________________________________"

cd ../
srun -n $SLURM_NTASKS python X3_parameter_studies.py 1
