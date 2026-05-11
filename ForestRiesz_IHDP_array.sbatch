#!/bin/bash
#SBATCH --job-name=fr_ihdp
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00
#SBATCH --mem=30000
#SBATCH --array=1-10
#SBATCH -o logs/slurm-fr-ihdp-%A_%a.out
#SBATCH -e logs/slurm-fr-ihdp-%A_%a.err
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

SHARD=$((SLURM_ARRAY_TASK_ID - 1))

echo "Running ForestRiesz_IHDP.py for shard ${SHARD} ${FORESTRIESZ_IHDP_ARGS:-}"
python ForestRiesz_IHDP.py \
	--mode simulate \
	--shard "${SHARD}" \
	--n-shards 10 \
	--n-jobs "${SLURM_CPUS_PER_TASK}" \
	${FORESTRIESZ_IHDP_ARGS:-}

# After all array tasks finish, run this once to create aggregate outputs:
# python ForestRiesz_IHDP.py --mode summarize --aggregate-shard-start 0 --aggregate-shard-end 9
