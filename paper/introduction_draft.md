# 1 Introduction (draft)

Citations are author-year in brackets; arXiv ids are in the reference list, which contains only papers that appear in the literature review (lit_review.md). [CHECK] marks a detail the literature review could not verify online or that must be confirmed against the locked methodology before submission.

## 1.1 The practical question

Consider a lab that has pretrained a decoder-only language model with S parameters, 25M ≤ S ≤ 1B, on web text and now holds a fixed budget for generating synthetic training data with an off-the-shelf instruction-tuned teacher. With the family fixed, the decision variable is the teacher's size T. Larger teachers write more accurate and more varied text but cost more per token, so the budget buys fewer of their tokens. Which T yields the largest improvement in the student, at equal synthetic tokens and at equal generation cost, and does the answer change with S? The mechanism is sequence-level distillation [Kim and Rush 2016]: the student is trained with cross-entropy on text the teacher wrote, never on the teacher's token probabilities.

## 1.2 Why prior answers come from neighbouring regimes

The literature answers several neighbouring questions but not this one. For logit distillation, Busbridge et al. [2025] fit a distillation scaling law in which a stronger teacher first helps and then hurts, and Zhang et al. [2025] report that the optimal teacher scales linearly with the student for students below 3B parameters; a student trained on synthetic text never sees those distributions and, with a different tokenizer, cannot. For instruction tuning of students of roughly 7B to 9B parameters [CHECK], Xu et al. [2024] find across twenty generators that stronger models are not stronger teachers, and Bansal et al. [2024] show that a weaker, cheaper generator wins when sampling compute is matched, because it produces about three times as many samples [CHECK]. For pretraining data, four generator-size studies report early saturation: rephraser family matters more than size [Pieler et al. 2024]; rephraser gains from 1B to 3B saturate by 8B for 3B and 8B students [Maini et al. 2025]; generators above roughly 8B do not yield better data for 100M to 3B students trained from scratch on mixtures, with no per-student optimum reported [Kang et al. 2025; CHECK]; and rephrasing gains stop at a 1B generator, at one student size [Niklaus et al. 2026; CHECK]. The SmolLM team reports that 70B-class generators did not improve on Mixtral-8x7B for Cosmopedia [Ben Allal et al. 2024].

These are literature claims about other channels or regimes: logit rather than sequence transfer, students of 7B or more, from-scratch mixtures rather than a shared pretrained base, rephrasing rather than open-ended generation, or one student size. Together they make the qualitative outcome predictable (larger is not always better; returns saturate between roughly 1B and 8B) but do not say where the curve bends for a 25M or a 1B student, whether the bend moves with S, whether the logit-channel relation between optimal teacher and student size survives when the student never sees teacher probabilities, or what each teacher's tokens cost. The capacity gap itself is not at issue [Cho and Hariharan 2019; Mirzadeh et al. 2020], and same-family teachers of 1.5B and 7B can be nearly indistinguishable to a small student [Li et al. 2026], so the study must be powered for small within-family effects.

## 1.3 What is held fixed and what varies

Five students (25M, 100M, 250M, 500M, 1B parameters) share one architecture family, one 32k tokenizer, one optimizer and one base corpus (FineWeb-Edu). Each is trained once with a warmup-stable schedule to a stable-phase checkpoint θ*(S); this places the treatment in the annealing phase, where small-model recipes put their highest-quality data [Hu et al. 2024]. Every condition is a branch from θ*(S) running the same linear-decay-to-zero phase over D₂ = 800M tokens: 600M treatment tokens plus 200M replay of already-seen base shards, identical across conditions. The treatment file is the only input that differs. Control arms fill it with repeated base data (C0), fresh held-out web text (C1, the primary comparator) or human-written question-answer text in the same format (C1b). Treatment arms fill it with synthetic text from a Qwen2.5-Instruct teacher of size T ∈ {1.5B, 3B, 7B, 14B, 32B, 72B}, plus a 0.5B extension and one cross-family anchor. All teachers write under one frozen pool of 4.0M decontaminated prompts, one system prompt, one decoding configuration, the same BF16 precision and teacher-agnostic filters with no quality judge; each student receives exactly 600M student-tokenizer tokens per teacher, subsampled by whole examples with identical per-category quotas. Generation cost is measured (GPU-seconds, energy, dollars, FLOPs), and a nested sweep over D_syn ∈ {75M, 150M, 300M, 600M} in a sub-grid supports an iso-cost comparison [Bansal et al. 2024].

In one sentence: with θ*(S), the phase-2 protocol Π₂ and the generation protocol fixed, τ_S(T) = E[Y(Π₂(θ*(S), F_T)) − Y(Π₂(θ*(S), F_real))] is the average effect of replacing 600M fresh web tokens with 600M synthetic tokens from teacher T in the annealing phase of a size-S student, and Δ_S(T, T′) = τ_S(T) − τ_S(T′) is the total effect of teacher identity, read as teacher size within the fixed family, conditional on the base corpus and state, D₂, the replay fraction, the schedule and the tokenizer.

## 1.4 Pre-registered hypotheses

Outcomes are a chance-corrected mean accuracy P_cc over a floor-gated benchmark set identical across S, and held-out negative log-likelihood on human-written instruction text. The analysis plan was frozen before the first full-study run; each hypothesis is stated so that it can fail.

H1 (monotonicity): P_cc(S, T) increases with T at every S. Test: the linear-in-log T term against per-S intercepts, and per-S isotonic regression with a permutation null. Literature expectation: false as a strict statement.

H2 (saturation): above some T_sat(S), doubling T raises P_cc by less than δ = 1.0 point (sensitivity 0.5). Test: one-sided negative curvature, and per-S non-inferiority of μ(72B) − μ(14B) against δ. Expectation: true, with T_sat between roughly 3B and 14B.

H3 (intermediate optimum): at some S, a teacher T₀ < 72B beats the 72B teacher. Test: a four-condition interior-optimum rule on a quadratic fit in log T, confirmed by a contrast on held-back seeds not used to select T₀. Expectation: plausible for 25M to 100M.

H4 (a relation between T* and S): the argmax over T moves with S. Test: an S × T interaction test; if at least three sizes have an interior optimum, a weighted regression of log T* on log S, reported descriptively with its confidence interval. Expectation: exploratory, possibly unidentifiable if the curve is flat above 3B.

H5 (cost versus performance optimum): the teacher maximising gain per generation dollar differs from the teacher maximising gain. Test: the marginal efficiency of the last upgrade into the performance-optimal teacher lies, with its confidence interval, below the average efficiency of the cheapest teacher. Expectation: nearly certain; the contribution is its magnitude as a function of S.

## 1.5 Contributions, stated as questions this design can answer

1. For each student size, is the teacher-size curve of sequence-level synthetic annealing data interior, saturating, monotone or flat, at equal tokens and against a matched-real control, with seed-based confidence intervals?
2. Does the linear relation between optimal teacher and student size reported for logit distillation [Zhang et al. 2025; Busbridge et al. 2025] transfer to a channel in which the student never sees teacher probabilities?
3. Where, as a function of S, do the performance-optimal and the cost-optimal teacher diverge when generation cost is measured rather than approximated by FLOPs, and how does the iso-cost view [Bansal et al. 2024] change the ranking?
4. Does teacher size retain any effect once the student's base-model likelihood on the teacher's text, the teacher's correctness on a verifiable subset and output diversity are controlled, in the spirit of the compatibility measure of Xu et al. [2024]?

All configurations, the frozen prompt pool, manifests and run-level results are released.

## 1.6 What the study cannot establish

The results are conditional on one teacher family, one prompt pool and one generation protocol; they do not separate teacher size from post-training recipe, and small instruct models may themselves have been trained on outputs of larger ones [CHECK for Qwen2.5]. Within one family, size and capability are nearly collinear. Five student sizes and six teacher sizes support at most a descriptive relation between T* and S with wide intervals, not a scaling law; "synthetic scaling laws" and "distillation scaling laws" already name other channels [Qin et al. 2025; Busbridge et al. 2025]. Single-generation synthetic data with the real base retained is not the model-collapse setting [Gerstgrasser et al. 2024; Schaeffer et al. 2025]. The design is off-policy [Agarwal et al. 2024] and says nothing about from-scratch synthetic pretraining or early mixing beyond a cross-check at the two smallest sizes. That a smaller teacher can produce a better student is not a new finding; the study measures where and by how much it happens in this regime.

## References

- Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Ramos, S., Geist, M., Bachem, O. 2024. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024. arXiv:2306.13649.
- Bansal, H., Hosseini, A., Agarwal, R., Tran, V. Q., Kazemi, M. 2024. Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling. ICLR 2025. arXiv:2408.16737.
- Ben Allal, L., et al. 2024. Cosmopedia: how to create large-scale synthetic data for pre-training; and SmolLM: blazingly fast and remarkably powerful. Hugging Face blog posts. [CHECK author list]
- Busbridge, D., Shidani, A., Weers, F., Ramapuram, J., Littwin, E., Webb, R. 2025. Distillation Scaling Laws. ICML 2025. arXiv:2502.08606.
- Cho, J. H., Hariharan, B. 2019. On the Efficacy of Knowledge Distillation. ICCV 2019. arXiv:1910.01348.
- Gerstgrasser, M., Schaeffer, R., et al. 2024. Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data. arXiv:2404.01413.
- Hu, S., et al. 2024. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. COLM 2024. arXiv:2404.06395.
- Kang, F., Ardalani, N., Kuchnik, M., Emad, Y., Elhoushi, M., Sengupta, S., Li, S., Raghavendra, R., Jia, R., Wu, C.-J. 2025. Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls. EMNLP 2025. arXiv:2510.01631.
- Kim, Y., Rush, A. M. 2016. Sequence-Level Knowledge Distillation. EMNLP 2016. arXiv:1606.07947.
- Li, Y., Yue, X., Xu, Z., Jiang, F., Niu, L., Lin, B. Y., Ramasubramanian, B., Poovendran, R. 2025. Small Models Struggle to Learn from Strong Reasoners. Findings of ACL 2025. arXiv:2502.12143.
- Li, Z., Zuo, Y., He, Y., Zhang, K., Xiao, Q., Qian, X., Yu, S., Gao, Z., Yang, Z., Liu, Z., Ding, N. 2026. Rethinking On-Policy Distillation of LLMs: Phenomenology, Mechanism, and Recipe. arXiv:2604.13016.
- Maini, P., et al. 2025. BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining. arXiv:2508.10975.
- Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., Ghasemzadeh, H. 2020. Improved Knowledge Distillation via Teacher Assistant. AAAI 2020. arXiv:1902.03393.
- Niklaus, J., Yamaguchi, A., Štefánik, M., Penedo, G., Kydlíček, H., Bakouch, E., Tunstall, L., Beeching, E., Frere, T., Raffel, C., von Werra, L., Wolf, T. 2026. How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data. arXiv:2604.13977.
- Pieler, M., et al. 2024. Rephrasing natural text data with different languages and quality levels for LLM pre-training. NeurIPS 2024 ENLSP workshop. arXiv:2410.20796.
- Qin, Z., Dong, D., et al. 2025. Scaling Laws of Synthetic Data for Language Models. arXiv:2503.19551.
- Schaeffer, R., Kazdan, J., Arulandu, A. C., Koyejo, S. 2025. Position: Model Collapse Does Not Mean What You Think. arXiv:2503.03150.
- Xu, Z., Jiang, F., Niu, L., Lin, B. Y., Poovendran, R. 2024. Stronger Models are NOT Stronger Teachers for Instruction Tuning. NAACL 2025. arXiv:2411.07133.
- Zhang, C., Li, D., Song, Y., Ye, Y., Gao, Y., Hu, X. 2025. Towards the Law of Capacity Gap in Distilling Language Models. ACL 2025. arXiv:2311.07052.
