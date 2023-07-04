#!/usr/bin/env bash
#SBATCH --job-name=comfetch-cifar    # sets the job name if not set from environment
#SBATCH --time=20:00:00    # how long you think your job will take to complete; format=hh:mm:ss
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
module load cuda/11.4.4
source ../../../../cmlscratch/marcob/environments/compressed/bin/activate

#Example of running a 
mpirun -n 4 python main.py --cr 0.25
