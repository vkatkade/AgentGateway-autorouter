# Agentic evals: τ³-Banking and SWE-bench Verified

`eval/run_eval.py` checks single prompts against labels. This directory evaluates the router the way Agent
Gateway will see it: **agent loops**, where one task makes many model calls and the router decides on every
call, so a single task can mix Argon and Hydrogen turns.

| Benchmark | Tasks | Score | Agent harness |
|---|---|---|---|
| τ³-Banking (`banking_knowledge`) | all 97, 3 trials each | DB / action reward (no LLM judge) | `tau2 run` |
| SWE-bench Verified | 100, pinned sample | the official test-based grader | mini-swe-agent + `swebench` |

> **Status.** Everything here is unit-tested offline (`tests/test_agentic.py`), and the pieces were checked
> against the real tools (tau2's `--dry-run` accepts the generated command, mini-swe-agent merges the generated
> config, a real LiteLLM client works against the proxy). It has **not** been run against live Vertex or with
> Docker yet. Do the pilot ([below](#pilot-before-the-full-run)) first.

## How it works

```
 tau2 / mini-swe-agent                        eval/agentic/proxy.py                      Vertex AI
 (LiteLLM, OpenAI shape)   ──/v1/chat/completions──▶  convert → apply arm ──generateContent──▶ gemini-4-argon
   x-eval-task: task_042                              │ router arm: ext_proc to the            gemini-3.8-flash
   x-eval-trial: 1                                    │ deployed router (as Agent Gateway)
                                                      └─▶ calls.jsonl (one record per call)
```

Neither benchmark knows about ext_proc, so a small proxy sits where the gateway would. It converts the
OpenAI request to Vertex `generateContent`, applies the experiment **arm**, calls Vertex with your ADC, and
converts back. For the router arms it runs the same exchange as `client/app.py`: request headers → one buffered
body → response headers, then applies the router's `:path` and body mutations. A router that is unreachable
fails open, as `failOpen: true` would.

**Attribution.** The harness LLM arguments are global per process, so the runners start **one harness process
per task** (per task and trial for τ³-Banking) and give each its own `x-eval-task` / `x-eval-trial` headers.
Every call is therefore tied to its task without guessing.

### Arms

Run every arm on the same tasks with the same grader, sampler settings and user simulator.

| Arm | What it does | Answers |
|---|---|---|
| `argon` | every call → Argon, the harness's own thinking settings | quality ceiling, cost baseline |
| `hydrogen` | every call → Hydrogen | cost floor |
| `argon_medium` | every call → Argon, `thinkingLevel` MEDIUM | does the router beat a fixed thinking level? |
| `router` | the router picks model and thinking level | **the system under test** |
| `router_no_thinking` | router's model choice; the harness's thinking settings restored | ablation of the thinking-level logic (PR #3) |

Model IDs and thresholds come from the same environment variables as the router (`ARGON_MODEL`,
`HYDROGEN_MODEL`, …); keep them identical to the deployed router's.

### Router backends

| `--router` | Use |
|---|---|
| `remote` | **Headline numbers.** The deployed Cloud Run router, over ext_proc/gRPC. From outside the VPC use the Cloud Run URL: `--router-target <host>:443 --router-tls --router-auth id_token` (needs `roles/run.invoker`). From a VM in the VPC use the internal LB: `--router-target 10.120.0.2:80`. |
| `inprocess` | The policy runs in the proxy (`GEMINI_API_KEY` needed). Fast iteration, and it records `classifier_ms`, `p_deep` and `thinking_source` per call, which the router does not return in headers. |

## Quick start

```bash
pip install -r requirements-dev.txt        # proxy, router protos, tests
python -m pytest tests/test_agentic.py     # offline, no keys

gcloud auth application-default login      # Vertex calls (or: export VERTEX_ACCESS_TOKEN=$(gcloud auth print-access-token), ~1 hour, no refresh)
export PROJECT_ID=my-project VERTEX_LOCATION=global
```

### τ³-Banking

tau2 needs Python 3.12–3.13 and its own environment (it cannot share one with `tau2-bench` v0.x):

```bash
git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench && uv sync --extra knowledge
# On Python 3.13, if `tau2 --help` fails with "No module named 'audioop'":  uv pip install audioop-lts
cd ..

python -m eval.agentic.run_tau3 --arm router --out runs --start-proxy \
    --router-target agw-autorouter-XXXX-uc.a.run.app:443 --router-tls --router-auth id_token \
    --tau2-repo ./tau2-bench --tau2-cmd "uv run --project ./tau2-bench tau2" \
    --user-llm vertex_ai/gemini-3.8-flash --user-llm-args '{"temperature":0,"vertex_location":"global"}' \
    --trials 3 --parallel 4
```

* The **user simulator** is a fixed model that is *not* routed, so it is identical in every arm. Pick it once.
* Banking tasks are scored on database state and required actions, so no judge model is involved.
* Retrieval is `bm25` (offline). Keep it identical across arms.
* Add `--limit 10` for a pilot, `--dry-run` to see the exact `tau2 run` command.

### SWE-bench Verified

Docker is required (agent sandbox and grader). Run on a VM, not Cloud Run:

```bash
pip install mini-swe-agent swebench

python -m eval.agentic.run_swebench --arm router --out runs --start-proxy \
    --router-target agw-autorouter-XXXX-uc.a.run.app:443 --router-tls --router-auth id_token \
    --parallel 4 --grade-workers 4 --step-limit 100
```

The 100 instance IDs are pinned in `swebench_verified_100.json` (proportional by repo, then by difficulty
label; seed 20261116; regenerate with `python -m eval.agentic.sample_swebench`). Proportional allocation leaves
out the two tiniest repos (seaborn, flask) and the three ">4 hours" tasks; the file records the exact
composition next to the population's. `--skip-grade` stops after the agent runs, `--grade-only` re-grades.

### All arms, then the report

```bash
for arm in argon hydrogen argon_medium router router_no_thinking; do
  python -m eval.agentic.run_tau3 --arm $arm --out runs --start-proxy ...      # same flags every time
done
python -m eval.agentic.analyze runs/tau3_banking-* runs/swebench_verified-* \
    --baseline argon --prices eval/agentic/prices.json --out runs/report.md
```

Runs resume: finished (task, trial) pairs are skipped on re-run (`--redo-errors` retries errored ones).

## Results format

One directory per run, `runs/<benchmark>-<arm>/`:

| File | Contents |
|---|---|
| `manifest.json` | `RunManifest`: arm, router mode/target, models, router env (thresholds, no secrets), task IDs, harness versions |
| `calls.jsonl` | one `CallRecord` per LLM call: models requested/sent, thinking level asked/sent, router outcome, latency, Vertex status, token usage |
| `tasks.jsonl` | one `TaskResult` per (task, trial): `success`, `score`, `termination`, `duration_s`, `error` |
| `harness/`, `grading/`, `proxy.log` | raw logs and artifacts |

Schemas: `schema.py` (dataclasses) and generated JSON Schemas in `schemas/` (`python -m eval.agentic.schema
--write`; a test keeps them in sync). `error` on a `TaskResult` means the harness failed (proxy or Vertex
error, grader error); it is **not** a task failure, and analysis excludes it.

## Reading the report

`analyze.py` reports, per arm: success rate, **quality retained** vs the baseline, paired Δ success, cost and
token savings, pass^k (banking), and routing behaviour (share routed to Hydrogen, thinking levels sent,
fail-open and classifier-error rates, router latency, Vertex errors).

* **Paired, on clean tasks.** Quality is compared task by task on the tasks that ran cleanly in both arms; CIs
  are 95% paired-bootstrap over tasks, and the sign test is exact (McNemar's when outcomes are binary).
* **Noise.** With 97–100 tasks one success rate carries about ±10 points. Only large effects are detectable.
  A sensible claim is "retains ≥ 95% of Argon quality at a clear cost saving", read from the CI's lower bound.
  SWE-bench Verified is also public and likely contaminated, so compare arms to each other, not to leaderboards.
* **Cost.** No prices are baked in. Copy `prices.example.json` to `prices.json` and fill in real per-million
  prices; thinking tokens are billed as output. Without it you get tokens only.
* **Timeouts hide in savings.** A classifier timeout fails open (the request passes through unchanged), so a
  high `classifier errors` rate means the router saved less than it could; compare `router ms` to
  `DECISIONS_TIMEOUT_MS`.
* **Remote mode** records the router's reason and round-trip time but not `p_deep` / `thinking_source`; use
  `--router inprocess` or the router's Cloud Logging for those.

## Pilot before the full run

Run 10 tasks per benchmark on arms `argon` and `router` (`--limit 10`) and check:

1. **Mixed-model turns work.** The proxy re-attaches Gemini thought signatures by tool-call ID, but a signature
   minted by one model may not be accepted by the other. Look at `signature_retry` in `calls.jsonl` and at
   Vertex errors. If routing mid-task breaks function calling, try `--signature-policy drop` or `dummy`; the
   retry-with-dummy fallback uses a placeholder value (`skip_thought_signature_validator`) that is **unverified**.
2. **Tool schemas and model name are accepted.** The proxy trims JSON Schema to the subset Vertex accepts;
   confirm no 400s. Confirm tau2 / LiteLLM also accept the custom `openai/agent` model name (the proxy ignores
   it and uses the arm), and that mini-swe-agent's tool-call parsing works with the proxy's responses.
3. **Attribution.** Every record in `calls.jsonl` has a real `task_id` (not `unknown`).
4. **Budget.** Tokens and dollars per task × tasks × arms × trials. Decide the trial count from this.
5. **Grader.** The SWE-bench report is read from `grading/*.<run-id>.json` (`resolved_ids`); confirm the file name
   with your installed `swebench` version.

## Files

```
schema.py          CallRecord / TaskResult / RunManifest, JSONL io, JSON Schema generation
schemas/           generated JSON Schemas
arms.py            the five arms; in-process router backend; apply_arm()
convert.py         OpenAI chat-completions <-> Vertex generateContent, thought-signature cache
extproc_client.py  ext_proc client for the deployed router (+ identity-token auth)
proxy.py           the OpenAI-compatible proxy (Flask); VertexClient with retries
common.py          run directories, proxy subprocess, manifests
sample_swebench.py seeded stratified sampler; swebench_verified_100.json is its pinned output
run_tau3.py        τ³-Banking runner (one tau2 process per task and trial)
run_swebench.py    SWE-bench runner (one mini-swe-agent process per instance) + grading
analyze.py         paired comparison and report
prices.example.json
```
