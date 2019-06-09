#!/usr/bin/env bash
#SBATCH -A C3SE2017-1-8
#SBATCH -p hebbe
#SBATCH -J ConvGrainSize
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 0-03:00:00

. /apps/new_modules.sh
module load intel

JULIA_PATH=$SNIC_NOBACKUP/julia/


$JULIA_PATH/julia jobrun.jl

# End script
