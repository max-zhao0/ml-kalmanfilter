#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -J generate_pairs
#SBATCH --mail-user=max_zhao@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00

#OpenMP settings:
#export OMP_NUM_THREADS=1
#export OMP_PLACES=threads
#export OMP_PROC_BIND=true
export HDF5_USE_FILE_LOCKING=FALSE

#run the application:
#applications may performance better with --gpu-bind=none instead of --gpu-bind=single:1 
module load python/3.9-anaconda-2021.11;
conda activate maxz_env
srun python /global/homes/m/max_zhao/mlkf/telescope32/generate_pairs.py
