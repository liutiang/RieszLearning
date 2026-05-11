#!/bin/bash
#SBATCH --job-name=fr_bhp                   # Set a name for your job
#SBATCH --partition=short                   # Slurm partition to use
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --time=0-03:00                      # Time limit in D-HH:MM
#SBATCH --mem=30000                         # Memory limit
#SBATCH --array=1-300                       # 300 tiny jobs: 10 seeds x 6 designs x 5 xfit/mult configs
#SBATCH -o logs/slurm-fr-bhp-%A_%a.out
#SBATCH -e logs/slurm-fr-bhp-%A_%a.err
#SBATCH --mail-type=NONE                    # NONE, BEGIN, END, FAIL, ALL
#SBATCH --mail-user=ivytgliu@uw.edu         # Email for notifications

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

VENV_PATH="${SLURM_SUBMIT_DIR}/.venv"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
	echo "Missing virtual environment at ${VENV_PATH}."
	echo "Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirment.txt"
	exit 1
fi

source "${VENV_PATH}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

echo "Running ForestRiesz_BHP.py for task ${TASK_INDEX} ${FORESTRIESZ_ARGS:-}"
python ForestRiesz_BHP.py \
	--mode simulate \
	--task-index "${TASK_INDEX}" \
	--n-jobs "${SLURM_CPUS_PER_TASK}" \
	${FORESTRIESZ_ARGS:-}

# After all array tasks finish, run ForestRiesz_BHP_summarize.sbatch once to create aggregate outputs.
