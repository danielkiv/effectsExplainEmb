#!/bin/bash

#SBATCH --job-name=ssi
#SBATCH --time=24:00:00
#SBATCH --mem=192g
#SBATCH -n 24
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=dkiv2@illinois.edu

# Load the pyton environment
echo "Loading conda environment"
source /data/keeling/a/dkiv2/dkiv2/miniconda3/bin/activate
conda activate night

cmds=(
    # "python run_ml_exps.py"
    # "python voting_start.py"
    # "python explainable_spatialeffects_embedding_test.py"
    # "python dumb.py"
    # "python embeddingsRunTorchSpatial.py"
    "python embeddingsRun.py"
)

for ((i = 0; i < ${#cmds[@]}; i++))
do
    # Print the command and run it
    echo "Running command: ${cmds[$i]}"
    ${cmds[$i]}
done

echo "Experimental run complete. Script closing."

# End the script
exit