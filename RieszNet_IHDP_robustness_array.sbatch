#!/bin/bash
#SBATCH --job-name=rn_ihdp_robust
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00
#SBATCH --mem=30000
#SBATCH --array=1-10
#SBATCH -o logs/slurm-rn-ihdp-robust-%A_%a.out
#SBATCH -e logs/slurm-rn-ihdp-robust-%A_%a.err
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SHARD=$((SLURM_ARRAY_TASK_ID - 1))

echo "Running RieszNet_IHDP.py robustness for shard ${SHARD} ${RIESZNET_IHDP_ROBUSTNESS_ARGS:-}"
python RieszNet_IHDP.py \
	--mode robustness \
	--shard "${SHARD}" \
	--n-shards 10 \
	--torch-num-threads "${SLURM_CPUS_PER_TASK}" \
	${RIESZNET_IHDP_ROBUSTNESS_ARGS:-}
