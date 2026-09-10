# 03. Novelty Assessment

Source: doc 04 (literature review, ~65 papers; novelty matrix in §6). This document states what the study can and cannot claim as new. Verification caveat: several 2025–2026 papers were checked only through secondary sources from the drafting environment; items tagged [UNVERIFIED] in doc 04 must be confirmed before the related-work section is final.

## 3.1 What the nearest prior work already establishes

| Claim | Where it is already shown | Consequence for this study |
|---|---|---|
| A larger teacher is not always a better teacher | Vision KD (Cho & Hariharan 2019; Mirzadeh et al. 2020); logit KD of LMs (Busbridge et al. 2025; Zhang et al. 2025; Lu & Liu 2026); instruction tuning (Xu et al. 2024); reasoning SFT at matched compute (Bansal et al. 2024); rephrasing generators (Pieler et al. 2024; BeyondWeb 2025; Kang et al. 2025; Niklaus et al. 2026); SmolLM/Cosmopedia generator comparison | Must be cited as established; the study **confirms or fails to confirm** it in a new regime, it does not discover it |
| Generator size saturates early for rephrasing-style synthetic pretraining data | BeyondWeb (1B→3B gain, ≈8B saturation); Kang 2025 (> ~8B no better); Niklaus 2026 (no gain beyond 1B, 30× cheaper) | Prior expectation is early saturation; the study's value is the S-dependence and the controls |
| The optimal teacher grows with student size (logit channel, < 3B) | Zhang et al. 2025 (linear T*(S)); Busbridge et al. 2025 (optimum in teacher-loss space) | H4 is a **transfer test** from the logit channel to the sequence-level channel, not a new law |
| Cheaper teachers can be compute-optimal | Bansal et al. 2024 (iso-FLOP); Niklaus 2026 | H5 is expected; contribution = measured magnitude vs S with real hardware costs |
| Best teacher depends on the student ("compatibility") | Xu et al. 2024 (CAR); GRACE; PerSyn; SCAS | The learnability mediator (S4) adopts this framing rather than claiming it |

## 3.2 What is unaddressed, as far as the review could verify

1. **A controlled teacher-size × student-size grid for sequence-level synthetic data at pretraining-scale students (25M–1B)** with one post-trained teacher family, one frozen generation protocol, equal student-tokenizer tokens, a matched-real-token control (C1) and a human-Q&A format control (C1b). Kang 2025 is closest but uses from-scratch mixtures and rephrasing/textbook data; Niklaus 2026 sweeps generator size for rephrasing at (apparently) one student size; Xu 2024 has the design at 7–9B across mixed families.
2. **Whether the logit-channel T*(S) results transfer to the sequence-level channel**, where the student never sees teacher probabilities.
3. **The divergence between T*(S) and T$(S) as a function of S with measured generation cost** (GPU-seconds, energy, dollars) on stated hardware, including the non-linearity of realised cost in T (batching and KV-cache limits give a ≈ 21× throughput spread against a 47× FLOP spread).
4. **A pre-registered mediation analysis** (student-base NLL of teacher data; teacher correctness on a verifiable set; diversity; length) testing whether "size" retains explanatory power once compatibility is controlled.
5. A **WSD-branching annealing design** in which every condition branches from the same stable-phase checkpoint into an identical decay, so that teacher contrasts are within-base paired comparisons and controls C0/C1/C1b/C_T form a clean factorial. This is a methodological contribution of modest size; the components (WSD, annealing-phase data injection, replay) are all established.

## 3.3 Compact novelty matrix

Rows: elements of this study. Columns: the five closest works. Cells: done / partial / not done (full ten-column matrix in doc 04 §6).

| Element | Busbridge 2025 | Bansal 2024 | Xu 2024 | Kang 2025 | Niklaus 2026 |
|---|---|---|---|---|---|
| Sequence-level synthetic data (not logits) | not done | done | done | done | done |
| Students 25M–1B, S axis | done (logit) | not done | not done | partial (100M–3B) | not done |
| Single instruct family, 6–7 sizes | done (own teachers) | not done | not done | partial | partial |
| Equal-token primary control + matched-real control | done / not done | not done | partial | done / not done | done / not done |
| Human Q&A format control (C1b) | not done | not done | not done | not done | not done |
| Iso-cost co-analysis | partial (FLOPs) | done | not done | partial | partial |
| Measured GPU-s / $ / kWh cost | not done | not done | not done | not done | partial |
| T*(S) with identifiability rule | done | not done | not done | not done | not done |
| Mediation (learnability, correctness, diversity) | partial | not done | partial (CAR) | not done | not done |

## 3.4 Claims the paper may and may not make

**May claim (if supported by the pre-registered tests):** the shape of the teacher-size transfer curve at each S in this family and regime; whether an interior optimum exists and is confirmed on held-out seeds; whether the curve shape depends on S; the measured cost–benefit frontier and the divergence between T* and T$; which data properties mediate the teacher effect; the value of synthetic annealing data relative to matched real tokens and to human Q&A text.

**Must not claim:** "synthetic scaling laws", "distillation scaling laws", or any "law"; being first to show that a smaller teacher can be a better teacher; T* ∝ S^α as a law (only a descriptive relation with CI, if ≥ 3 sizes have an identifiable optimum); generality beyond Qwen2.5-Instruct, beyond the prompt mixture, or beyond the annealing regime; separation of teacher size from post-training recipe or from capability within one family; anything about model collapse; that equal-token is the uniquely correct control (iso-cost is co-reported).

## 3.5 Assessment

The project is incremental in its qualitative direction and genuinely unaddressed in its regime, controls, and cost measurement. Its defensible contribution is a careful measurement, not a discovery. The title is chosen accordingly (doc 00 §0.1).
