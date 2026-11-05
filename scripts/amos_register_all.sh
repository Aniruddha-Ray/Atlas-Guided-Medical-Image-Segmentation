#!/bin/bash
# Registers the 10 AMOS atlases onto every other AMOS case (all 272 minus the atlases), 3 shards in parallel.
# Resumable: cached targets are skipped. Logs: logs/amos_reg_shard<i>.out
cd "$(dirname "$0")/.."
D=../Dataset061_AMOS
for i in 0 1 2; do
  ( for sp in train val; do
      PYTHONIOENCODING=utf-8 python -W ignore scripts/register_targets.py --data-dir $D \
        --atlas-library cache_amos/atlas_library/atlas_library.npz --selection atlas/atlas_selection_amos.json \
        --out-dir cache_amos --split $sp --num-shards 3 --shard-index $i --eval-gt --continue-on-error
    done ) >> logs/amos_reg_shard$i.out 2>&1 &
done
wait
echo "ALL SHARDS DONE" >> logs/amos_reg_shard0.out
