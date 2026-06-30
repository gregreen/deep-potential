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
  --punk-width 64 \
  --punk-depth 5 \
  --phi-width 64 \
  --phi-depth 5 \
  --batch-size 256 \
  --n-epochs 512 \
  --run-dir runs/test

python3 -m partial_obs.benchmark_partial_obs \
  --n-eval 4096 \
  --run-dir runs/test
