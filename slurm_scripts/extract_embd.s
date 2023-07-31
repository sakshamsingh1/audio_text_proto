#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=20GB
#SBATCH --output=output_%j.out

local=/tmp/$USER/local
mkdir -p $local

singularity \
    exec \
    --overlay /scratch/sk8974/envs/zs_test/zs_test.ext3:ro \
    /scratch/sk8974/envs/dcase/dcase.sif \
    /bin/bash -c "
source /ext3/env.sh
python3 -u extract_embd.py --model_type clap --dataset_name us8k
"