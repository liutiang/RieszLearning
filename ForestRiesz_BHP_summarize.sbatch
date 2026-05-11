#!/bin/bash
#SBATCH --job-name=fr_bhp_sum
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-02:00
#SBATCH --mem=30000
#SBATCH -o logs/slurm-fr-bhp-summarize-%j.out
#SBATCH -e logs/slurm-fr-bhp-summarize-%j.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=ivytgliu@uw.edu

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

echo "Running ForestRiesz_BHP.py summarize ${FORESTRIESZ_SUMMARIZE_ARGS:-}"
python ForestRiesz_BHP.py --mode summarize ${FORESTRIESZ_SUMMARIZE_ARGS:-}
