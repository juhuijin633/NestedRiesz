#!/bin/bash
#
# Main paper simulations only:
#   - Linear DGP + truncated logistic, propensity in [0.1, 0.9]
#   - Nonlinear DGP + truncated adversarial, propensity in [0.3, 0.7]
#
# Submit:
#   cd Simulation_1_TimeTreatment && sbatch code/submit.sh

#SBATCH --job-name=timetreatment_main
#SBATCH -p serial_requeue
#SBATCH -t 02:00:00
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --array=0-2999
#SBATCH -o logs/job_%a.out
#SBATCH -e logs/job_%a.err

module load python/3.10.9-fasrc01
module load R/4.4.3-fasrc01

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate riesz

export R_LIBS_USER="${R_LIBS_USER:-${HOME}/R/library}"
mkdir -p "${R_LIBS_USER}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ -f "${SUBMIT_DIR}/code/run_sim.py" ]]; then
  APP_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/run_sim.py" ]]; then
  APP_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Could not find run_sim.py. Submit from Simulation_1_TimeTreatment/ or its code/ subfolder." >&2
  exit 1
fi

cd "${APP_DIR}"
mkdir -p "${SUBMIT_DIR}/logs"

# -----------------------------------------------------------------------
# Index mapping:
# 2 DGPs x 3 N values x 500 iterations = 3000 total jobs
#
# DGP 0 (linear, truncated_logistic [0.1, 0.9]):
#   0    - 499  : N=500
#   500  - 999  : N=1000
#   1000 - 1499 : N=2000
#
# DGP 1 (nonlinear, truncated_adv [0.3, 0.7]):
#   1500 - 1999 : N=500
#   2000 - 2499 : N=1000
#   2500 - 2999 : N=2000
# -----------------------------------------------------------------------

N_VALUES=(500 1000 2000)
TMAX=500

DGP_IDX=$(( SLURM_ARRAY_TASK_ID / (${#N_VALUES[@]} * TMAX) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (${#N_VALUES[@]} * TMAX) ))
N_IDX=$(( REMAINDER / TMAX ))
ITER=$(( REMAINDER % TMAX ))

N_VAL=${N_VALUES[$N_IDX]}

if [ $DGP_IDX -eq 0 ]; then
    python "${APP_DIR}/code/run_sim.py" $ITER $N_VAL truncated_logistic 0.1 0.9
else
    python "${APP_DIR}/code/run_sim_nonlinear.py" $ITER $N_VAL truncated_adv 0.3 0.7
fi
