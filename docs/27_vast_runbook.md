# 27. Vast.ai Runbook (hybrid, money-safe)

This is the exact procedure for running the C4 study on vast.ai, written so that once you fund a
balance and give the assistant your vast.ai API key, it can provision, run, monitor, and tear down
the experiment without wasting money. Read the "Network requirement" and "Money safety" sections
before funding anything.

## Network requirement (read this first)

The assistant drives vast.ai through the vast.ai HTTPS API. A Claude Code remote environment only
allows that if its **egress network policy permits `vast.ai`**. The default "trusted" policy
(package registries and GitHub only) **blocks `vast.ai`, `console.vast.ai`, and `huggingface.co`**,
so from such an environment the `vastai` CLI cannot reach the API at all (every call fails with a
`403 CONNECT` policy denial), and the assistant cannot provision or monitor.

To let the assistant run this end to end, one of the following must be true:

1. **Recommended.** Run the assistant in a Claude Code environment whose network policy allows
   `vast.ai` (a custom allowlist that includes `vast.ai` and `*.vast.ai`, or a full-egress policy).
   Network policy is chosen when the environment is created; see
   https://code.claude.com/docs/en/claude-code-on-the-web. The vast.ai box that runs the study has
   its own unrestricted network, so Hugging Face and model downloads there are unaffected — only
   the assistant's own session needs `vast.ai` reachable.
2. **Alternative.** Run the six commands below yourself on your own computer (install the CLI with
   `pip install vastai`, `vastai set api-key <KEY>`), and paste the outputs back. The scripts are
   self-contained and money-safe; the assistant will guide each step and read your pasted results.

Everything else in this repository is ready now regardless of which path you choose.

## The topology decision, and why it is a single box

Vast.ai is a marketplace of independent machines. Unlike RunPod, it does **not** give you one
network volume that many separate spot instances can all mount at once. The fleet design in
`docs/26` assumes a shared volume, so on vast.ai we map it to the arrangement that is actually
robust there:

**One multi-GPU instance. Local disk is the shared volume. One GPU-pinned fleet worker per GPU.**

* The file-locked job queue (`synscale/fleet.py`) lives on that instance's local disk, where file
  locks work perfectly, so all the parallelism, dependency ordering, generation sharding, and
  preemption-requeue logic runs unchanged. `scripts/fleet_local.sh` starts one worker per GPU,
  each pinned with `CUDA_VISIBLE_DEVICES`.
* An 8-GPU H100 box does the ~1,230 GPU-hours of C4 in about 6 to 7 days of wall-clock; a 4-GPU
  box in about 13 days. Cost is the same total dollars either way.
* One instance means one thing to monitor and one thing to destroy. There is no cross-machine
  data movement and no dependence on vast.ai network volumes.

"Hybrid" on a single box means: rent it **on-demand** for reliability (recommended, and the
money-safe default), or **interruptible** (`--bid`) for roughly a third off with the risk that the
whole box stops if you are outbid. Either way the run is checkpointed and resumable, so a stop
loses minutes, not the run.

## What you provide

1. **A funded vast.ai balance.** The full C4 hybrid run is about **$1,900 USD (~$2,650 CAD)**, plus
   roughly 25 percent for preemption re-runs, so fund around **$2,400 to $2,800 USD** if you want it
   to finish without topping up. You can start with less: your balance is a hard cap (see below).
2. **Your vast.ai API key.** Create it at https://console.vast.ai/manage-keys/ and either run
   `vastai set api-key <KEY>` in this session or paste it so the assistant can. It is stored only
   in this session's container.
3. **A Hugging Face token with Llama-3.x access** (for C4's second teacher family). Request access
   on the Llama model pages at huggingface.co, then pass the token as `--hf <HF_TOKEN>`. Qwen-only
   profiles do not need it. The token is uploaded to the run instance as a file and used only to
   download the gated weights.

The assistant already has the `vastai` CLI installed and the whole toolkit under `scripts/vast/`.

## The staged sequence (what the assistant runs)

```bash
# 0. one time, after you provide the key
vastai set api-key <YOUR_KEY>

# 1. preflight: confirm auth, balance, an SSH key, and that H100 offers exist
./scripts/vast/preflight.sh

# 2. MONEY GATE: rent one cheap single GPU, run the smoke profile end to end on real CUDA,
#    verify it produces an index, then destroy the box. Costs about $1-5.
./scripts/vast/smoke.sh

# 3. only if smoke PASSES: launch the main run on one multi-GPU H100 box
./scripts/vast/launch.sh c4 --gpus 8 --hf <HF_TOKEN>            # on-demand (reliable)
#   or, cheaper and interruptible:
./scripts/vast/launch.sh c4 --gpus 8 --hf <HF_TOKEN> --bid 12.0

# 4. monitor (one snapshot per call; the assistant schedules these as check-ins)
./scripts/vast/monitor.sh

# 5. the moment it reports DONE: pull results and destroy the box
./scripts/vast/teardown.sh
```

The smoke gate is the single most important money-saver: it catches "the code does not run on this
GPU/driver/vLLM" for a few dollars, before any H100 budget is spent. The assistant will not launch
step 3 unless step 2 passes.

## Money safety, built in

* **Nothing is created without a timed wait-and-verify.** Every `create` is paired with a poll loop
  that has a timeout and **destroys the instance** if it reaches `exited`, `offline`, `unknown`, or
  never reaches `running`. A stuck box cannot bill silently.
* **Smoke always self-destructs** via an EXIT trap, pass or fail.
* **Your balance is the hard cap.** If credit runs out mid-run, vast.ai stops the box (disk is
  preserved) instead of letting it run up a debt. Add credit and re-launch the fleet to resume from
  the checkpoints and queue on disk.
* **Teardown fetches results first, then destroys, then lists any instance still billing** so you
  can see at a glance that spending has stopped.
* The assistant is instructed to tear the box down the moment the study is DONE, and to never leave
  an instance running idle.

## Monitoring across days, honestly

The study runs on the vast.ai box, which is independent of this assistant session. The assistant
monitors by waking on a schedule and running `monitor.sh` (status, phase, queue progress, log tail,
balance, DONE marker). Two honest caveats:

* This assistant's own container is ephemeral and can be reclaimed. If that happens, **the vast.ai
  run keeps going** (it is detached on the box), but automated monitoring pauses until the session
  is resumed. It does not cost you anything extra; it just means a check-in might be late.
* Because of that, the box is built to run fully unattended: the onstart hook installs and launches
  the fleet on its own, writes progress to `/workspace/status.txt` and `/workspace/run.log`, and
  writes `/workspace/DONE` with a summary when finished. You can always run `monitor.sh` yourself.
* The balance cap is the backstop that protects you even in the worst case where monitoring stops
  entirely: spending cannot exceed what you funded.

## If something fails

* **Smoke fails.** The assistant dumps the install and run logs from the box before destroying it,
  diagnoses, fixes the code, and re-runs smoke. No H100 budget is touched until smoke is green.
* **A teacher download 404s or is gated.** Confirm Llama access on your HF account, or swap the
  `-AWQ` repo for a `-GPTQ-Int4` equivalent in the profile. Qwen repos are open.
* **The box is outbid (interruptible).** It moves to `stopped` with disk intact. Raise the bid to
  resume, or the assistant re-launches on-demand; the fleet continues from checkpoints.
* **"No space left on device."** The launcher defaults to a 600 GB disk; raise it with `--disk`.

## Files

```
scripts/vast/preflight.sh    auth + balance + ssh key + offer check
scripts/vast/smoke.sh        the $1-5 real-GPU money gate (auto-destroys)
scripts/vast/launch.sh       provision one multi-GPU box + start the run
scripts/vast/onstart.sh      unattended box setup + fleet launch (uploaded to the box)
scripts/vast/monitor.sh      one status snapshot (RUNNING | DONE | STOPPED | GONE)
scripts/vast/teardown.sh     fetch results, destroy, confirm billing stopped
scripts/vast/common.sh       shared helpers (all money-safe: timeouts + auto-destroy)
scripts/fleet_local.sh       on-box: one GPU-pinned worker per GPU (the parallel engine)
```
