#!/bin/bash
#
# Submit either way:
#   cd Simulation_2_DiD && sbatch code/submit.sbatch   # recommended
#   cd Simulation_2_DiD/code && sbatch submit.sbatch
#
# SLURM log files (logs/out_*.out) are written under the directory you submit from.
# Simulation outputs always go under Simulation_2_DiD/results/.
#
# After the array finishes:
#   python code/collect_results.py

#SBATCH --job-name=did_sim
#SBATCH --array=0-8
#SBATCH -o logs/out_%a.out
#SBATCH -e logs/err_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jinhopark@college.harvard.edu
#SBATCH -t 4-00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --partition=intermediate

set -eo pipefail

module load python/3.10.13-fasrc01
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate riesz

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ -f "${SUBMIT_DIR}/code/RUN.py" ]]; then
  APP_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/RUN.py" ]]; then
  APP_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Could not find RUN.py. Submit from Simulation_2_DiD/ or Simulation_2_DiD/code/." >&2
  exit 1
fi

cd "${APP_DIR}"
mkdir -p "${SUBMIT_DIR}/logs" "${APP_DIR}/results"

python "${APP_DIR}/code/RUN.py"
