# 02. Design Overview (one page)

```
                    FineWeb-Edu train split (one seeded shuffle; decontaminated)
                                    │ nested prefixes
          ┌─────────────────────────┼─────────────────────────┐
     5B tokens                 10B tokens                 20B tokens
   s025m / s100m / s250m         s500m                      s1b
          │  warmup → constant LR (WSD stable phase)          │
          ▼                                                   ▼
     θ*(S): stable-phase checkpoint (weights + Adam state + loader position)
          │
          │  every arm branches here; identical linear decay to 0 over D₂ = 800M tokens
          │  = 600M treatment + 200M replay of seen shards (same shards, same slots)
          ├── C0   : treatment = re-seen base shards            ("nothing new")
          ├── C1   : treatment = fresh held-out web             (matched real; primary comparator)
          ├── C1b  : treatment = human StackExchange Q&A        (format control)
          ├── T0.5B … T72B : treatment = teacher pool subsampled to 600M student tokens
          └── D-sweep: treatment = {75,150,300}M synthetic + fresh web to 600M
                                                  │
   Qwen2.5-Instruct {0.5,1.5,3,7,14,32,72}B ──────┘  one prompt pool (4.0M), one system prompt,
   BF16, vLLM, one decoding config, teacher-agnostic filters, union-drop decontamination,
   equal student-tokenizer tokens, Alpaca-style rendering, full-sequence loss
                                                  │
                                                  ▼
   lm-eval 0.4.13 zero-shot: Tier A {SciQ, ARC-E, PIQA, LAMBADA} + gated Tier B {OBQA, SIQA, HellaSwag}
   + held-out NLL (human instruction; real ID/OOD) + task loss + secondary suite + format probe
                                                  │
                                                  ▼
   P_cc (chance-corrected mean) → models M0–M4 → per-S T* classification → STE, T$(λ), iso-cost
```

**Grid.** 5 students × 7 teachers (6-rung core primary) + 3 controls per student; 3 phase seeds (5 at the ends and around the argmax; C1 at 5); pilot at 25M/100M with 3 base × 3 phase seeds; D-sweep at {100M, 1B} × {3B, 14B, 72B} × 3 levels × 2 seeds.

**Identification.** Δ_S(T,T′) is a within-base paired contrast; the only file that differs between two arms at the same S and seed is the treatment file. Cross-S inference is conditional on one base run per size (base replicates at 25M/100M bound the base-seed variance).

**Outputs.** Transfer curves per S with fitted forms and bootstrap bands; per-S classification {interior, saturating, monotone-unsaturated, flat}; descriptive T*(S) relation; cost-per-token by teacher; STE and T$(λ) frontiers; mediator analysis; contamination audit; full cell table.

**What decides whether the full grid runs.** Pilot rules P1–P6 (doc 05, doc 12 §7): detectable synthetic-vs-C1 effect, noise ceiling, floor gate, teacher spread, dose response, pipeline integrity.
