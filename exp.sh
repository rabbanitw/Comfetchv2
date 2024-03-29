#!/usr/bin/env bash
#SBATCH --job-name=comfetch-cifar    # sets the job name if not set from environment
#SBATCH --time=01:00:00    # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=high    # set QOS, this will determine what resources can be requested
#SBATCH --partition=dpart
#SBATCH --ntasks=4
#SBATCH --mem 32gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:2
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE
#SBATCH --nodelist=cml21

module purge
module load mpi
# module load cuda/11.1.1
# source ../../../../cmlscratch/marcob/environments/compressed/bin/activate

# Example of running a 10% (weight reduced by 90%) sketch with 200 comm rounds
mpirun -n 4 python main.py --cr 0.1 --name run1-sketch --seed 101 --epochs 200
mpirun -n 4 python main.py --cr	0.1 --name run2-sketch --seed 102 --epochs 200
mpirun -n 4 python main.py --cr	0.1 --name run3-sketch --seed 103 --epochs 200
