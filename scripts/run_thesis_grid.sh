#!/usr/bin/env bash
# Thesis experiment grid (2 architectures x 2 backbones x 2 pre-trainings) on 3 seeds.
# Kept for compatibility: the grid is now the campaign configs/campaigns/thesis_grid.yaml.
#
#   bash scripts/run_thesis_grid.sh                 # run or resume the grid
#   bash scripts/run_thesis_grid.sh --dry-run       # what would run, disk and time estimates
#   bash scripts/run_thesis_grid.sh --seeds 0       # single seed
#
# Results: runs/campaigns/thesis_grid/report.md, MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
set -euo pipefail
exec rsseg campaign configs/campaigns/thesis_grid.yaml "$@"
