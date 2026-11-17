"""Agentic evaluation of the auto-router: tau3-banking and SWE-bench Verified.

See eval/agentic/README.md. Modules:

  schema          result records (calls.jsonl, tasks.jsonl, manifest.json) + JSON Schemas
  arms            the experiment arms (always-Argon, always-Hydrogen, router, ablations)
  convert         OpenAI chat-completions <-> Vertex generateContent
  extproc_client  plays the gateway's part against the deployed router (ext_proc over gRPC)
  proxy           OpenAI-compatible proxy: benchmark harness -> router -> Vertex
  sample_swebench seeded, stratified 100-task sample of SWE-bench Verified
  run_tau3        runs tau3-banking through the proxy (one tau2 process per task/trial)
  run_swebench    runs SWE-bench Verified through mini-swe-agent + the proxy, then grades
  analyze         paired comparison of arms: quality, cost, routing behaviour, reliability
"""
