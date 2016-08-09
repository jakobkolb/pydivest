#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --qos=short
#SBATCH --job-name=X4_scan
#SBATCH --output=X4_scan_%j.out
#SBATCH --error=X4_scan_%j.err
#SBATCH --time=0-12
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --mail-type=ALL,TIME_LIMIT_80
#SBATCH --mail-user=jakob.j.kolb@gmail.com

module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

echo "_______________________________________________"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "_______________________________________________"

cd ../
srun -n $SLURM_NTASKS python X4_phase_transition.py
