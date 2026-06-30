python3 -m partial_obs.mock_data --n-samples 8192 --dim-spec xyvz

python3 -m partial_obs.run_partial_obs \
  --dim-spec xyvz \
  --n-samples 32768 \
  --symmetry spherical \
  --punk-type gaussian \
  --train-pobs \
  --pobs-batch-size 1024 \
  --pobs-epochs 512 \
  --pobs-width 64 \
  --pobs-depth 5 \
  --punk-width 32 \
  --punk-depth 3 \
  --phi-width 32 \
  --phi-depth 3 \
  --batch-size 1024 \
  --n-epochs 128 \
  --run-dir runs/test \
  --resume

python3 -m partial_obs.benchmark_partial_obs \
  --n-eval 4096 \
  --run-dir runs/test
