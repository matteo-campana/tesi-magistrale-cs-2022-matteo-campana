#!/usr/bin/env bash
# Thesis experiment grid (2 architectures x 2 backbones x 2 pre-trainings) on N seeds.
#
#   bash scripts/run_thesis_grid.sh            # 3 seeds
#   SEEDS="0" bash scripts/run_thesis_grid.sh  # single seed
#
# Prerequisites: `rsseg prepare -c configs/base.yaml` (raw files in data/download/Potsdam).
# Results: MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) and runs/*/report.json.
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}

for arch in r18 r34; do
  if [ ! -f "runs/dino_${arch}_backbone.weights.h5" ]; then
    rsseg pretrain -c "configs/ssl/dino_${arch}.yaml"
  fi
done

for seed in $SEEDS; do
  for exp in maresunet maresunet_vq; do
    for arch in r18 r34; do
      for pre in imagenet dino; do
        rsseg train -c "configs/experiments/${exp}_${arch}_${pre}.yaml" \
          --set train.seed="$seed" tracking.tags.seed="'$seed'" tracking.tags.grid=thesis
      done
    done
  done
done

rsseg report runs/ --out reports/thesis_grid.md
