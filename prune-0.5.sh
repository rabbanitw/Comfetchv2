#!/usr/bin/env bash
#SBATCH --job-name=prune0.5   # sets the job name if not set from environment
#SBATCH --time=10:00:00    # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --ntasks=4
#SBATCH --mem 32gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:2
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module purge
module load mpi
module load cuda/11.1.1
source ../../../../cmlscratch/marcob/environments/compressed/bin/activate

#Example of running a 50% (weight reduced by 50%) sketch with 200 comm rounds
mpirun -n 4 python main_prune.py --cr 0.5 --name run1-prune --seed 101 --epochs 200
mpirun -n 4 python main_prune.py --cr	0.5 --name run2-prune --seed 102 --epochs 200
mpirun -n 4 python main_prune.py --cr	0.5 --name run3-prune --seed 103 --epochs 200