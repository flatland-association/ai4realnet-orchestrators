#!/bin/bash
set -euxo pipefail
source /home/conda/.bashrc
source activate base
conda activate flatland-rl

export PYTHONPATH=$PWD

flatland-trajectory-generate-from-policy $@

