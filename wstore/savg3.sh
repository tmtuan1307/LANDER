#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Oct 10 2023 16:30:58 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="cfl1"

# The project ID which this job should run under:
#SBATCH --account="punim2101"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per node:
#SBATCH --mem=24576

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-20:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi
# Run the job from the directory where it was launched (default)

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

module load Anaconda3/2022.10
export CONDA_ENVS_PATH=/data/gpfs/projects/punim2101/anaconda3/envs
eval "$(conda shell.bash hook)"
conda activate p39

wandb online
python main.py --wandb=1 --group=5tasks_cifar100 --method=nayeravg --r=1e-1 --tasks=5  --beta=0 --nums=8000 --kd=1 \
--exp_name=nayeravg_c30_fr1_ln25r1e1bn1e3g40 --type -1 --bn 1e-3 --num_users 5 --fr 1 --syn_round 10 --swp 0 --lte_norm 10 \
--com_round 30 --g_steps 40