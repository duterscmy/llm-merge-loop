# CoMe setup on Arrhenius

## Current state

- CoMe is cloned locally at `third_party/CoMe` and pinned by inspection to commit
  `7bb695adba4b69df45020e281348e99739824c39`.
- The local clone has two reproducible patches under `patches/come/`. They fix
  a malformed progress-log call and a missing `logging` import in the PPL
  evaluator without changing the merging algorithm.
- `third_party/` is intentionally ignored so the nested Git repository does not
  pollute this repository's history.
- The ARM64 Conda environment is named `come` and now lives at
  `/nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come`.
- It was created on a GH200 compute node by cloning the existing `log-prob`
  environment, preserving its working `torch==2.6.0+cu126` build, then installing
  the pure-Python versions in `env/come-requirements.txt`.
- The environment passed `pip check`, CUDA detection, CoMe method imports, and
  the custom LLaMA implementation import on a GH200.
- Slurm job `3161609` verified the cloned nobackup environment, including a
  Mistral forward pass. Job `3161615` then removed only the old home-directory
  copy and installed `~/.condarc` entries that prefer nobackup for environments
  and package caches. Job `3161692` added an activation hook that also directs
  Pip's cache to `/nobackup/.../personal/mingyu/cache/pip`.
- Meta rejected access to `meta-llama/Llama-2-7b-hf`, so the first pilot uses
  the public Apache-2.0 `mistralai/Mistral-7B-v0.1` checkpoint instead. It is a
  7B-class, 32-layer base model supported by CoMe.
- Mistral-7B-v0.1 is downloaded at `models/Mistral-7B-v0.1` as safetensors
  (about 14 GB on disk). Slurm job `3155184` verified it on an ARM64 GH200:
  `MistralForCausalLM`, 32 physical layers, 7,241,732,096 parameters, a
  `(1, 6, 32000)` logits tensor, and 13.52 GiB of allocated CUDA memory.

## Important CoMe source audit findings

- The README advertises `laco`, but the registered implementation raises
  `NotImplementedError`.
- The bundled MKA implementation computes a similarity matrix but currently
  merges the last two remaining layers at every compression step.
- Therefore CoMe should remain a third-party candidate implementation rather
  than replacing this repository. Its maintained method is `concat_merge`.

## Verify the environment

Run from the project root:

```bash
sbatch script/verify_come_env.slurm
```

Then inspect `logs/come-verify-<job-id>.log`.

Do not activate this ARM64 environment on the login node. In Slurm scripts,
activate it by absolute path:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /nobackup/proj/disk/ehpc-reg-2026r01-278/personal/mingyu/conda/envs/come
```

## Download and verify Mistral-7B-v0.1

The model is public and does not require Hugging Face authentication. Submit the
download and CoMe loading smoke test from the project root:

```bash
sbatch script/download_mistral.slurm
```

The model is stored at `models/Mistral-7B-v0.1` and is ignored by Git. Re-running
the job resumes or verifies the existing files, then checks the architecture,
layer count, parameter count, and a short forward pass with `use_cache=False`
on a GH200. The download is restricted to safetensors plus configuration and
tokenizer files, so it does not fetch a duplicate `.bin` copy of the weights.

## First concat-merge smoke experiment

Run the reproducible 32-to-31-layer merge with:

```bash
sbatch script/run_come_concat_smoke.slurm
```

Slurm job `3155294` completed on a GH200. With eight shuffled WikiText-2 rows
used only as a minimal calibration smoke test, CoMe merged original layers 26
and 27 with weights 0.5074 and 0.4926. The saved model has 31 layers and
7,023,620,096 parameters, a 3.01% reduction from the 32-layer baseline. It was
successfully reloaded by standard Transformers for forward and generation.

Compare baseline and merged-model WikiText-2 perplexity with:

```bash
sbatch script/eval_come_concat_smoke.slurm
```

Job `3155354` evaluated 163 non-overlapping 2048-token test blocks. Baseline
PPL was 5.2521 and merged-model PPL was 5.6017, a 6.66% relative increase. This
is a pipeline validation, not a final quality claim: the calibration sample is
deliberately tiny and the result needs comparison with layer deletion and the
proposed looping variants.

## Paper-configured Mistral branch and merge + loop pilot

The public Mistral checkpoint is also an evaluated CoMe paper model. The first
formal target is its 9.04% setting: 32 to 29 physical layers, 256 WikiText-2
calibration samples, two-layer progressive merging, `p=1`, and minimum retained
importance ratio `rho=0.97`. Submit it with:

```bash
sbatch script/run_come_mistral_9pct.slurm
```

Job `3155784` completed the 32-to-29 merge with groups `[23,24]`,
`[25,26]`, and `[27,28]`. It measured WikiText-2 PPL 6.5997 and C4 PPL
9.9994. The paper reports 6.52 and 10.04 respectively for Mistral at 9.04%
parameter sparsity, so the reproduction differs by about +1.2% and -0.4%.

The loop implementation lives outside CoMe under `loop/`. It converts CoMe's
final `merge_info` mapping into a logical execution plan and inserts repeated
references to the same physical module. It never deep-copies looped layers and
asserts that the parameter count is unchanged. KV caching is disabled while a
loop plan is active because a shared attention module has one physical cache
index but appears at multiple logical depths.

The smoke comparison uses only eight WikiText-2 blocks:

```bash
sbatch script/eval_come_loop_smoke.slurm
```

Job `3155738` validated all five execution modes and confirmed that repeated
module references do not change the physical parameter count. On this small
slice, PPL was 4.8696 for dense, 5.1938 for the 31-layer merged model, 5.6049
for merge-aware looping, 5.3639 for the uniform control, and 5.3901 for the
seeded random control. This is a negative preliminary signal for naive looping,
not a conclusion: the underlying 31-layer smoke checkpoint used only eight
calibration rows and CoMe's default `rho=0`, unlike the paper configuration.

The full 9% comparison uses every complete 2048-token WikiText-2 test block and
evaluates dense, merged, merge-aware loop, uniform-loop control, and seeded
random-loop control:

```bash
sbatch script/eval_come_mistral_9pct_loop.slurm
```

Job `3155837` completed all 163 blocks. Dense PPL was 5.2521 and the 29-layer
CoMe model was 6.5997. At the same 6,587,396,096 physical parameters, restoring
32 logical executions gave 7.9125 for merge-aware looping, 7.4008 for the
uniform control, and 8.1200 for the seeded random control. Thus naive
merge-aware repetition beats this random draw but does not recover the merged
model's quality and is worse than the uniform control. This is a useful
negative result and motivates testing residual-scaled or trained recurrence
rather than treating mergeability as sufficient for direct repetition.

## Downstream benchmark reproduction

The vendored evaluation harness was reduced to the tasks used in CoMe's
Mistral table and wired to accept the in-memory shared-module loop schedule.
The task registry covers ARC-Easy, ARC-Challenge, HellaSwag, OpenBookQA, PIQA,
WinoGrande, and all 57 MMLU subjects. PIQA uses the schema-compatible Parquet
mirror `lighteval/piqa` because the original script-based Hub repository is not
readable by the pinned Datasets/Hugging Face Hub pair.

Job `3161675` passed a 10-example PIQA loop smoke, job `3161682` passed a joint
six-task smoke, and job `3161693` passed a 5-shot MMLU smoke. Full six-task
zero-shot evaluation for dense, merged, merge-aware, uniform, and random
variants is Slurm array `3161685`. The corresponding 57-subject MMLU 5-shot
array is `3161710`, with an `afterok:3161685` dependency.
