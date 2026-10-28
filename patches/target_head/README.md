# Selective lm-head and compact-sampling prototype

This patch is an inference-only optimization for the sparse position-selective
route. It keeps the full logits shape but computes only the Position range plus
EOS for Position rows and the Feature range for Feature rows. Apply it inside
the remote GaussianGPT checkout with:

```bash
python patches/target_head/apply_selective_lm_head.py
python -m py_compile model/gpt/nanochat_gpt.py model/gpt/position_selective_speculative.py
```

It must pass full-vs-selective logits checks, cache invariants, and paired
wall-clock benchmarks before being treated as a valid optimization.

For the follow-up compact-support optimization, apply the selective head patch
first and then run:

```bash
python patches/target_head/apply_compact_sampling.py
python -m py_compile model/gpt/position_selective_speculative.py
```

The compact patch samples only legal Position (+ EOS) or Feature ids and maps
the sampled local index back to the global vocabulary. It preserves the
existing top-p policy and Position residual correction; it is not a new
acceptance rule. The current endpoint evidence and its limitations are recorded
in `docs/project/EXPERIMENTS.md` under S6.
