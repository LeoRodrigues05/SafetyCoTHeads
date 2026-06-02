# Direction A — SHIPS slice configurations.
#
# Pre-registered in `docs/direction_a/prereg.md`.
#
# Pipeline (run in order):
#   1. 01-ships-discovery-llama31.yaml        run_attribution.py
#   2. 02-ships-discovery-r1distill.yaml      run_attribution.py
#   3. 03-baseline-gen-llama31-jbb.yaml       run_generation.py  (seeds 0,1)
#   4. 04-baseline-gen-r1distill-jbb.yaml     run_generation.py  (seeds 0,1)
#   5. 05-ships-ablation-llama31-jbb.yaml     run_generation.py  (seeds 0,1)
#   6. 06-ships-ablation-r1distill-jbb.yaml   run_generation.py  (seeds 0,1)
#   7. 07-trajectory-judge.yaml               run_trajectory.py  (per generation file)
#
# Each generation script call should set `decoding.seed` and the matching
# `seed` override (and optionally `output.dir`) per seed.
